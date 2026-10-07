# Your Mail

All your mail accounts in one search, on the blank screen: the university's, Gmail, any account that speaks IMAP. Buhera searches them where they are, opens a message without marking it read, and can keep your recent mail on disk so that a search says whether your mail mentions a thing at all — instead of handing back its best look-alike.

**Time:** 15 minutes, most of it setting up the first account.

**Before you start:** mail is read by the long-grass server on your own computer (`npm run dev` in `long-grass/`). The hosted site will not read mail: it cannot tell whose mail a visitor may see, so it answers "mail is read only on the machine this server runs on".

---

## 0. What mail does, and what it never does

| It does | It never does |
|---|---|
| search every account at once | send, reply, move, flag or delete anything |
| open a message (it stays unread) | show your password to the browser |
| keep recent mail as Markdown files on your disk, if you ask | upload your mail anywhere |

## 1. Your accounts

Accounts live in one file outside every repository: `~/.buhera/mail.json` (on Windows `C:\Users\<you>\.buhera\mail.json`; set `MAIL_ACCOUNTS_FILE` to put it elsewhere).

```json
{
  "dir": "~/.buhera/mail",
  "accounts": [
    { "id": "uni", "label": "university",
      "host": "<your university's IMAP host>", "user": "<your login>",
      "password_env": "MAIL_UNI_PASSWORD" },
    { "id": "gmail", "host": "imap.gmail.com", "user": "you@gmail.com",
      "password_env": "MAIL_GMAIL_PASSWORD" }
  ]
}
```

- `password_env` names the variable that holds the password. Put the passwords in `long-grass/.env.local`, beside the other keys, then restart the server:

```text
MAIL_UNI_PASSWORD=…
MAIL_GMAIL_PASSWORD=…
```

- **Gmail** needs an app password (Google account → Security → 2-Step Verification → App passwords), not your normal one. Buhera notices `imap.gmail.com` and searches it with Gmail's own search, across All Mail.
- **University mail**: your IT department's pages name the IMAP host. Port 993 and TLS are the default; set `"port"` and `"secure"` if yours differs. Exchange servers need IMAP switched on for your account.
- `"mailboxes": ["INBOX", "Sent"]` searches more than the inbox.

Now open the top edge and pick **mail**. Each account says whether it is ready, or exactly what is missing ("the variable MAIL_UNI_PASSWORD is not set on this server", "no user").

## 2. Searching

Write `mail` and then what you are looking for:

```
mail from:mara
```

On a test mailbox holding a few weeks of a colleague's messages about the lipid series, this gave:

```text
mail "from:mara" · 2 matches · uni 2
2026-10-02  ● Re: LARA run                    Mara Lind  uni  + keep on a plan
2026-09-21  ● LARA run for the lipid series   Mara Lind  uni  + keep on a plan
```

The dot marks unread mail. The search syntax is the one most mail programs share; every part must hold:

| Write | Finds mail |
|---|---|
| `from:anna`, `to:me`, `cc:lab` | with that in the sender, recipient or copy (part of a name or address is enough) |
| `subject:"plate layout"` | with that in the subject; quotes keep words together |
| `since:2026-09-01`, `before:2026-10-01` | written on or after / before that date |
| `is:unread`, `is:read`, `is:flagged` | by state |
| `in:Sent` | in that mailbox instead of the account's usual ones |
| `account:uni` | in one account only |
| any other words | containing every one of them, anywhere in the message |

`inbox` alone shows the newest mail of every account.

## 3. Reading

Click a subject. The message opens as a new frame — sender, recipients, date, attached files by name and size, and the text (HTML mail is turned into plain text). Opening it does **not** mark it read; Buhera only looks.

```text
LARA run for the lipid series
from  Mara Lind mara@uni.test
to    Kundai Sachikonye
date  21.9.2026, 09:12:00 · uni / INBOX · uid 1

Hi Kundai,

the LARA robot is free on Thursday. Can you send the plate layout for the PC 34:1 lipid series
before Wednesday? We need the blank positions too.
```

**+ keep on a plan** puts the message on an experiment or task; [Finding and Planning](./finding-and-planning) picks this up.

## 4. Keeping mail, so a search can say "no"

A live search answers with whatever matches, or nothing. It cannot tell you that your mail **covers** a thing — or that it plainly does not mention it, and that you should look elsewhere. For that, keep your recent mail on disk: on the mail page, set the number of days and press **sync**.

```text
kept the last 90 days of mail in ~/.buhera/mail
uni  5 new, 0 already kept
indexed 5 messages — searches now answer with a verdict.
```

Each message becomes one Markdown file, `<dir>/<account>/<mailbox>/<month>/…md`. Syncing again fetches only what is new. Now ask the kept mail:

```
dispatch("mail", { kind: "ask", query: "internal standard blanks" })
```

```text
covered — uni/INBOX/2026-10/…-5.md:1-8 contains every query term
uni/INBOX/2026-10/…-5.md:4-8   matched internal, standard, blanks   open the mail
  8 │ Thursday works. Bring the blanks & the internal standard.
```

The verdict comes first: **covered** (one message holds every word), **partial** (the words are there, never all in one message) or **declined** (your mail does not contain what you asked about). Under a declined verdict it says which words are missing from all your mail. How far a verdict lets you go is spelled out in [Finding and Planning](./finding-and-planning) §2.

Kept mail is plain text on your disk, readable by anyone who can read your files. Delete the folder to forget it.

Next: [Finding and Planning](./finding-and-planning).
