// interceptor-run: sandboxed execution CLI for long-grass's interceptor module.
//
// Takes a Rust source file, compiles it with rustc, runs the resulting binary
// under a wall-clock timeout with capped stdout/stderr, and prints exactly one
// JSON object to stdout describing the outcome. Mirrors the CLI-spawn contract
// long-grass's API routes already use for `spraypaint`/`purpose` (spawn a
// binary, parse one JSON object from stdout) so `interceptor-run.js` can spawn
// this exactly the same way.
//
// Usage:
//   interceptor-run --code-file <path> [--timeout-ms <n>]
//
// Output (always exactly one line of JSON on stdout, regardless of outcome):
//   { ok, stage: "compile"|"run", stdout, stderr, exit_code, elapsed_ms,
//     timed_out, truncated }

use serde::Serialize;
use std::env;
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const DEFAULT_TIMEOUT_MS: u64 = 10_000;
const MAX_OUTPUT_BYTES: usize = 2 * 1024 * 1024; // 2 MiB cap per stream

#[derive(Serialize)]
struct RunResult {
    ok: bool,
    stage: &'static str,
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    elapsed_ms: u128,
    timed_out: bool,
    truncated: bool,
}

fn emit(result: RunResult) -> ! {
    println!("{}", serde_json::to_string(&result).unwrap_or_else(|_| {
        r#"{"ok":false,"stage":"run","stdout":"","stderr":"failed to serialize result","exit_code":null,"elapsed_ms":0,"timed_out":false,"truncated":false}"#.to_string()
    }));
    std::process::exit(0);
}

fn cap(s: String) -> (String, bool) {
    if s.len() <= MAX_OUTPUT_BYTES {
        return (s, false);
    }
    let mut end = MAX_OUTPUT_BYTES;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    (s[..end].to_string(), true)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut code_file: Option<PathBuf> = None;
    let mut timeout_ms: u64 = DEFAULT_TIMEOUT_MS;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--code-file" => {
                i += 1;
                if i < args.len() {
                    code_file = Some(PathBuf::from(&args[i]));
                }
            }
            "--timeout-ms" => {
                i += 1;
                if i < args.len() {
                    timeout_ms = args[i].parse().unwrap_or(DEFAULT_TIMEOUT_MS);
                }
            }
            _ => {}
        }
        i += 1;
    }

    let code_file = match code_file {
        Some(p) => p,
        None => emit(RunResult {
            ok: false,
            stage: "compile",
            stdout: String::new(),
            stderr: "interceptor-run: --code-file is required".to_string(),
            exit_code: None,
            elapsed_ms: 0,
            timed_out: false,
            truncated: false,
        }),
    };

    let source = match fs::read_to_string(&code_file) {
        Ok(s) => s,
        Err(err) => emit(RunResult {
            ok: false,
            stage: "compile",
            stdout: String::new(),
            stderr: format!("interceptor-run: cannot read code file: {err}"),
            exit_code: None,
            elapsed_ms: 0,
            timed_out: false,
            truncated: false,
        }),
    };

    let work_dir = env::temp_dir().join(format!(
        "interceptor-run-{}-{}",
        std::process::id(),
        Instant::now().elapsed().as_nanos()
    ));
    if let Err(err) = fs::create_dir_all(&work_dir) {
        emit(RunResult {
            ok: false,
            stage: "compile",
            stdout: String::new(),
            stderr: format!("interceptor-run: cannot create work dir: {err}"),
            exit_code: None,
            elapsed_ms: 0,
            timed_out: false,
            truncated: false,
        });
    }

    let src_path = work_dir.join("snippet.rs");
    let bin_name = if cfg!(windows) { "snippet.exe" } else { "snippet" };
    let bin_path = work_dir.join(bin_name);

    if let Err(err) = fs::write(&src_path, &source) {
        let _ = fs::remove_dir_all(&work_dir);
        emit(RunResult {
            ok: false,
            stage: "compile",
            stdout: String::new(),
            stderr: format!("interceptor-run: cannot write source: {err}"),
            exit_code: None,
            elapsed_ms: 0,
            timed_out: false,
            truncated: false,
        });
    }

    let t0 = Instant::now();

    // --- Compile stage -----------------------------------------------------
    let compile = Command::new("rustc")
        .arg("-O")
        .arg("--edition=2021")
        .arg(&src_path)
        .arg("-o")
        .arg(&bin_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    let compile = match compile {
        Ok(o) => o,
        Err(err) => {
            let _ = fs::remove_dir_all(&work_dir);
            emit(RunResult {
                ok: false,
                stage: "compile",
                stdout: String::new(),
                stderr: format!("interceptor-run: failed to spawn rustc: {err}"),
                exit_code: None,
                elapsed_ms: t0.elapsed().as_millis(),
                timed_out: false,
                truncated: false,
            });
        }
    };

    if !compile.status.success() {
        let (stderr, truncated) = cap(String::from_utf8_lossy(&compile.stderr).to_string());
        let _ = fs::remove_dir_all(&work_dir);
        emit(RunResult {
            ok: false,
            stage: "compile",
            stdout: String::new(),
            stderr,
            exit_code: compile.status.code(),
            elapsed_ms: t0.elapsed().as_millis(),
            timed_out: false,
            truncated,
        });
    }

    // --- Run stage (polling timeout, std-only) ------------------------------
    let mut child = match Command::new(&bin_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(err) => {
            let _ = fs::remove_dir_all(&work_dir);
            emit(RunResult {
                ok: false,
                stage: "run",
                stdout: String::new(),
                stderr: format!("interceptor-run: failed to spawn compiled binary: {err}"),
                exit_code: None,
                elapsed_ms: t0.elapsed().as_millis(),
                timed_out: false,
                truncated: false,
            });
        }
    };

    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => break,
            Ok(None) => {
                if Instant::now() >= deadline {
                    timed_out = true;
                    let _ = child.kill();
                    let _ = child.wait();
                    break;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(_) => break,
        }
    }

    let mut stdout_buf = String::new();
    let mut stderr_buf = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout_buf);
    }
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_string(&mut stderr_buf);
    }
    let exit_code = child.try_wait().ok().flatten().and_then(|s| s.code());

    let (stdout, t1) = cap(stdout_buf);
    let (stderr, t2) = cap(stderr_buf);

    let _ = fs::remove_dir_all(&work_dir);

    emit(RunResult {
        ok: !timed_out && exit_code == Some(0),
        stage: "run",
        stdout,
        stderr,
        exit_code,
        elapsed_ms: t0.elapsed().as_millis(),
        timed_out,
        truncated: t1 || t2,
    });
}
