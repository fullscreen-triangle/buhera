/**
 * Kuramoto phase-lock for scheduler synchronisation (network-yield §swarm).
 *
 * K scheduler instances each carry a phase phi_i in [0, 2pi). The synchronisation
 * order parameter is
 *   R e^{i psi} = (1/K) sum_i e^{i phi_i},
 * with R = 1 perfect phase-lock and R = 0 maximal desynchronisation. Under
 * Kuramoto coupling K_c the system locks (R -> R_inf > 0) once K_c exceeds the
 * critical coupling K_c* = 2 sigma_omega / pi. A drop of R below 0.95 despite
 * adequate coupling is a first-class fault signal (partition or scheduler death).
 */
/** The phase-lock threshold: R >= 0.95 counts as locked. */
export declare const LOCK_THRESHOLD = 0.95;
/** Order parameter of a set of phases: magnitude R and mean phase psi. */
export declare function orderParameter(phases: ReadonlyArray<number>): {
    R: number;
    psi: number;
};
/** Critical coupling K_c* = 2 sigma_omega / pi for a frequency spread sigma_omega. */
export declare function criticalCoupling(sigmaOmega: number): number;
/**
 * A bank of Kuramoto oscillators (one per scheduler instance). Deterministic
 * integrator (explicit Euler); natural frequencies are supplied, not randomised,
 * so runs are reproducible (the contract leaves the integrator open, §13).
 */
export declare class KuramotoBank {
    private readonly omega;
    private phases;
    private readonly coupling;
    constructor(naturalFreqs: ReadonlyArray<number>, coupling: number, initialPhases?: ReadonlyArray<number>);
    /** Advance all phases by dt using the mean-field Kuramoto update. */
    step(dt: number): void;
    /** Run for `steps` steps of size `dt`. */
    run(steps: number, dt: number): void;
    order(): {
        R: number;
        psi: number;
    };
    isLocked(): boolean;
    currentPhases(): ReadonlyArray<number>;
}
//# sourceMappingURL=kuramoto.d.ts.map