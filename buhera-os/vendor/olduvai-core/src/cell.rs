//! Interval arithmetic for quantities that are never points.
//!
//! # ⭐ Why this exists, and why it is not a duplicate of [`crate::fusion`]
//!
//! [`crate::fusion`] answers *where is this participant*, and it answers it in one dimension
//! that happens to be two-dimensional: metres on a tangent plane. This module answers the same
//! question for every **other** quantity the exchange handles — a price, a journey time, a
//! tonnage, a moisture reading — and those are scalar, so they need their own algebra rather
//! than a borrowed one.
//!
//! The commitment is the same one, though, and it is worth stating in the form the source
//! framework states it:
//!
//! > Every quantity in the agent system is a BAND, never a point. A cell is an interval
//! > `[lo, hi]` whose width is a measured floor, not a chosen tolerance. When two cells overlap,
//! > the system declines to rank them rather than inventing a number to break the tie.
//!
//! That is [`crate::fusion::Estimate::rests_on_observation`] generalised. `fusion` refuses to
//! draw a position it did not observe; this refuses to *rank* two prices it cannot separate.
//!
//! # ⚠️ The gap this closes
//!
//! Before this module, olduvai had no way to say **"unpriced"**. A commodity with no quote was
//! an absent field, which meant the assistant either omitted it silently or the model supplied
//! a plausible number. [`Cell::unknown`] is the third option and the correct one: an interval
//! too wide to be separated from anything. It composes, it sorts, it simply never wins. So a
//! hop with no fare, or a grade with no market quote, can sit in a ranking as a live hypothesis
//! without a fabricated value being invented to hold its place.
//!
//! ⭐ That is `notes/29`'s empty dictionary given a type. A missing entry is not a hole to be
//! filled; it is a maximally wide entry that loses every comparison honestly.
//!
//! # ⚠️ The second gap: observations that never age
//!
//! A GPS fix taken six hours ago folds into [`crate::fusion`] with exactly the sigma it had when
//! it was taken. That is wrong, and it is wrong in the dangerous direction — stale evidence
//! keeps full authority. [`Cell::widen`] plus [`staleness_widening`] is the correction: evidence
//! ages, its cell widens, and eventually it stops being separated from the field and stops
//! deciding anything.
//!
//! ⭐ Note the shape of that. Nothing is evicted, and no rule says "discard after N minutes". A
//! stale claim dies by *becoming indistinguishable*, which is the same test used for everything
//! else here. One decision procedure, applied to freshness as well as to value.
//!
//! # Purity
//!
//! Pure arithmetic. No clock, no I/O, no policy. [`staleness_widening`] takes an age rather than
//! reading one, for the same reason [`crate::fusion::Estimate::update`] takes an observation's
//! timestamp rather than a clock: the fold over a log must produce the same bytes on every
//! machine, forever.

use serde::{Deserialize, Serialize};
use std::fmt;

/// The floor on any cell's half-width, in the cell's own units, expressed as a relative epsilon.
///
/// ⚠️ **Authored, and a guard rather than a physical constant** — the same standing as
/// [`crate::fusion::MIN_SIGMA_M`]. A source claiming a zero-width price is claiming a point
/// value, which the Price-Cell Theorem forbids for any bounded observer. We refuse it at
/// construction rather than let an unattainable claim propagate into a ranking where it would
/// win every separation test by fiat.
pub const MIN_HALF_WIDTH: f64 = f64::EPSILON;

/// What kind of quantity a cell measures.
///
/// ⚠️ **Carried on the cell rather than tracked by the caller**, and checked in every binary
/// operation. Composing a time cell with a price cell is a programming error, not a wide
/// result, and a system whose entire discipline is "do not invent numbers" cannot then silently
/// add minutes to dollars. This is [`crate::units::Unit`]'s argument applied to intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellUnit {
    /// A duration. Milliseconds.
    Millis,
    /// Money, in the minor unit of whatever currency the caller is working in.
    Money,
    /// Mass. Tonnes.
    Tonnes,
    /// Distance. Metres.
    Metres,
    /// A dimensionless proportion in `[0,1]`.
    Ratio,
    /// Anything else. ⚠️ Two `Other` cells are considered compatible, so this is an escape
    /// hatch that gives up the unit guard — reach for it only when the quantity genuinely has
    /// no dimension worth naming.
    Other,
}

impl fmt::Display for CellUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            CellUnit::Millis => "ms",
            CellUnit::Money => "money",
            CellUnit::Tonnes => "t",
            CellUnit::Metres => "m",
            CellUnit::Ratio => "ratio",
            CellUnit::Other => "generic",
        };
        f.write_str(s)
    }
}

/// Why a binary operation on two cells could not be performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellError {
    /// `hi < lo`, or a bound was NaN.
    NotAnInterval,
    /// A half-width was zero or negative, which asserts a point value.
    ZeroFloor,
    /// The two cells measure different quantities.
    UnitMismatch { left: CellUnit, right: CellUnit },
    /// An operation that needs at least one cell was given none.
    Empty,
}

impl fmt::Display for CellError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CellError::NotAnInterval => write!(f, "not an interval: hi < lo, or a bound was NaN"),
            CellError::ZeroFloor => write!(
                f,
                "half-width must be > 0: a zero floor asserts a point value, which no bounded \
                 observer attains"
            ),
            CellError::UnitMismatch { left, right } => {
                write!(f, "unit mismatch: {left} vs {right}")
            }
            CellError::Empty => write!(f, "no cells"),
        }
    }
}

impl std::error::Error for CellError {}

/// A quantity known to lie in `[lo, hi]`.
///
/// ⭐ There is no constructor that takes a single number. That is deliberate and it is the whole
/// point of the type: a caller holding a point value must say what its floor is before this
/// module will represent it, which forces the question "how well do you actually know that?"
/// at the boundary rather than three layers in.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Cell {
    pub lo: f64,
    pub hi: f64,
    pub unit: CellUnit,
}

impl Cell {
    /// A cell from explicit bounds.
    ///
    /// ⚠️ Rejects `hi < lo` and NaN. The comparison is written as `!(hi >= lo)` rather than
    /// `hi < lo` so that a NaN bound fails it — NaN compares false to everything, and the
    /// negated form is the one that catches it.
    pub fn new(lo: f64, hi: f64, unit: CellUnit) -> Result<Cell, CellError> {
        if !(hi >= lo) {
            return Err(CellError::NotAnInterval);
        }
        Ok(Cell { lo, hi, unit })
    }

    /// A cell from a midpoint and a half-width floor: `[mid - beta, mid + beta]`.
    ///
    /// ⭐ This is the same construction as [`crate::fusion::Observation::Fix`]'s `sigma_m` — a
    /// centre and an admitted error — and it is written the same way here because it *is* the
    /// same thing in one dimension.
    ///
    /// ⚠️ `beta` must exceed [`MIN_HALF_WIDTH`]. A zero floor asserts a point, and a point is
    /// exactly the claim this whole module exists to make unrepresentable.
    pub fn from_floor(mid: f64, beta: f64, unit: CellUnit) -> Result<Cell, CellError> {
        if !(beta > MIN_HALF_WIDTH) {
            return Err(CellError::ZeroFloor);
        }
        Cell::new(mid - beta, mid + beta, unit)
    }

    /// The unknown cell: infinitely wide, and fully a member of the algebra.
    ///
    /// ⭐ **The key move.** An unquoted commodity is not a missing value — it is a cell too wide
    /// to be separated from anything. It composes, it appears in rankings, it never leads. So no
    /// caller is ever forced to invent a number in order to have something to sort.
    ///
    /// ⚠️ Contrast with `Option<Cell>`, which is the design this replaces. An `Option` pushes the
    /// decision to every call site, and the easy thing to write at a call site is
    /// `unwrap_or(some_plausible_default)` — which is fabrication with a shrug. An infinite cell
    /// makes the honest handling the *default* handling.
    pub fn unknown(unit: CellUnit) -> Cell {
        Cell {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
            unit,
        }
    }

    /// Is either bound non-finite?
    pub fn is_unknown(&self) -> bool {
        !self.lo.is_finite() || !self.hi.is_finite()
    }

    /// `hi - lo`. Infinite for an unknown cell.
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// The midpoint, or `None` when there isn't one.
    ///
    /// ⚠️ `Option` rather than NaN. A caller that wants a single number out of a cell is doing
    /// something the type is designed to make them think about, so the unknown case is returned
    /// as a value they must handle rather than as a float that will quietly poison arithmetic
    /// three functions later.
    pub fn mid(&self) -> Option<f64> {
        if self.is_unknown() {
            None
        } else {
            Some((self.lo + self.hi) / 2.0)
        }
    }

    /// Widen by `delta` on each side.
    ///
    /// Used for ageing: see [`staleness_widening`]. An unknown cell is returned unchanged,
    /// because there is nothing wider to make it.
    pub fn widen(&self, delta: f64) -> Cell {
        if delta <= 0.0 || self.is_unknown() {
            return *self;
        }
        Cell {
            lo: self.lo - delta,
            hi: self.hi + delta,
            unit: self.unit,
        }
    }

    /// Multiply both bounds by a non-negative factor.
    pub fn scale(&self, k: f64) -> Cell {
        if k < 0.0 || self.is_unknown() {
            return *self;
        }
        Cell {
            lo: self.lo * k,
            hi: self.hi * k,
            unit: self.unit,
        }
    }

    /// Do these two cells share any point? Touching endpoints count.
    pub fn overlaps(&self, other: &Cell) -> bool {
        self.lo <= other.hi && other.lo <= self.hi
    }

    /// The distance between two disjoint cells; zero when they overlap.
    pub fn gap(&self, other: &Cell) -> f64 {
        if self.overlaps(other) {
            return 0.0;
        }
        if self.lo > other.hi {
            self.lo - other.hi
        } else {
            other.lo - self.hi
        }
    }

    /// Are these two cells distinguishable at floor `beta`?
    ///
    /// ⭐ **This is the decision procedure, and the reason it returns `bool` rather than an
    /// ordering matters.** Two cells are separated when their *gap* exceeds the floor — not when
    /// their midpoints differ. A midpoint comparison always produces an answer, including when
    /// the two quantities are indistinguishable given what was measured. This one can honestly
    /// say no.
    ///
    /// ⚠️ An unknown cell is never separated from anything, including from another unknown. That
    /// is correct: knowing nothing about two quantities is not grounds for ordering them.
    pub fn separated(&self, other: &Cell, beta: f64) -> bool {
        if self.is_unknown() || other.is_unknown() {
            return false;
        }
        self.gap(other) > beta
    }

    /// Two independent observations of the **same** quantity, combined.
    ///
    /// The truth lies in both bands, so it lies in the overlap. This is how a cell narrows, and
    /// it is the scalar analogue of [`crate::fusion::Estimate::update`] — with the difference
    /// that a Kalman update always produces a posterior, whereas an intersection can fail.
    ///
    /// ⚠️ Returns `Ok(None)` on disjoint inputs. **That is information, not an error**: two
    /// sources that cannot both be right have told you something real — one is stale, or one is
    /// about a different thing. Averaging them would destroy exactly that signal, which is why
    /// there is no fallback branch here that quietly returns a midpoint.
    pub fn intersect(&self, other: &Cell) -> Result<Option<Cell>, CellError> {
        if !units_compatible(self.unit, other.unit) {
            return Err(CellError::UnitMismatch {
                left: self.unit,
                right: other.unit,
            });
        }
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if hi < lo {
            return Ok(None);
        }
        Ok(Some(Cell {
            lo,
            hi,
            unit: self.unit,
        }))
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_unknown() {
            write!(f, "[unknown {}]", self.unit)
        } else {
            write!(f, "[{:.2}, {:.2}] {}", self.lo, self.hi, self.unit)
        }
    }
}

/// ⚠️ [`CellUnit::Other`] is compatible with itself only. Everything else must match exactly.
fn units_compatible(a: CellUnit, b: CellUnit) -> bool {
    a == b
}

/// Interval addition: the quantity that is the sum of several quantities.
///
/// ⭐ **This is what makes a delegation tree work without anyone writing error propagation.** A
/// parent holding three legs does not sum three estimates, it composes three bands, and the
/// result is wider than any child's. Uncertainty accumulates upward because interval addition
/// does it, not because a formula was applied.
///
/// ⚠️ If any part is unknown, the whole is unknown, and that is the honest answer rather than a
/// degenerate one: one unpriced leg makes a journey's total price genuinely unbounded.
pub fn compose(cells: &[Cell]) -> Result<Cell, CellError> {
    let first = cells.first().ok_or(CellError::Empty)?;
    let unit = first.unit;
    let mut lo = 0.0;
    let mut hi = 0.0;
    for c in cells {
        if !units_compatible(unit, c.unit) {
            return Err(CellError::UnitMismatch {
                left: unit,
                right: c.unit,
            });
        }
        lo += c.lo;
        hi += c.hi;
    }
    Cell::new(lo, hi, unit)
}

/// The worst part, not the sum of the parts.
///
/// ⭐ **Not every quantity accumulates along a path, and treating them all as if they did is a
/// real error rather than an approximation.** Time and money accumulate: three legs cost the sum
/// of three legs. Exposure to bad weather does not — a journey through three regions does not
/// experience the sum of three storms, it experiences the worst one, because that is the leg
/// that strands you. Summing would rank a long fair-weather route below a short blizzard, which
/// is backwards.
///
/// The parent takes the child with the highest upper bound and keeps that child's **whole band**,
/// not just its `hi`: the parent's uncertainty is the uncertainty of the leg that dominates it,
/// and collapsing to a point would assert a precision no child has.
///
/// ⚠️ Any unknown part makes the whole unknown. A leg nobody could forecast might be the bad one,
/// and treating it as benign would rank an unobserved route above an observed one — which is the
/// specific failure this crate exists to prevent.
pub fn compose_worst(cells: &[Cell]) -> Result<Cell, CellError> {
    let first = cells.first().ok_or(CellError::Empty)?;
    let unit = first.unit;
    for c in cells {
        if !units_compatible(unit, c.unit) {
            return Err(CellError::UnitMismatch {
                left: unit,
                right: c.unit,
            });
        }
        if c.is_unknown() {
            return Ok(Cell::unknown(unit));
        }
    }
    let worst = cells
        .iter()
        .fold(first, |w, c| if c.hi > w.hi { c } else { w });
    Ok(*worst)
}

/// What a ranking concluded.
///
/// ⭐ **A verdict, not a sorted list**, and that is the entire design. A sorted list always has a
/// first element, so a caller handed one will treat it as the answer whether or not the data
/// supported picking it. Making "I cannot separate these" a distinct variant means the caller
/// has to handle it.
///
/// This is the same shape as [`crate::foreman::Outcome`]'s decline and
/// [`crate::fusion::Estimate::rests_on_observation`]: every procedure here terminates in
/// convergence or in an honest decline that names what it could not distinguish.
#[derive(Debug, Clone, PartialEq)]
pub enum Ranking {
    /// One item, separated from every other by more than the floor.
    Convergent {
        /// Index into the input slice.
        leader: usize,
        /// Input indices, best first. Unknowns last.
        ranked: Vec<usize>,
    },
    /// No single item is separated from the rest.
    ///
    /// ⚠️ `contenders` is the point of this variant. A decline that says only "no" is unusable;
    /// one that names the classes it could not separate tells the caller what a further
    /// observation would have to distinguish.
    Declined {
        /// The indices that could not be separated from the leader.
        contenders: Vec<usize>,
        ranked: Vec<usize>,
    },
}

impl Ranking {
    /// The leading index, if the ranking converged.
    pub fn leader(&self) -> Option<usize> {
        match self {
            Ranking::Convergent { leader, .. } => Some(*leader),
            Ranking::Declined { .. } => None,
        }
    }

    /// Every index, best first.
    pub fn ranked(&self) -> &[usize] {
        match self {
            Ranking::Convergent { ranked, .. } | Ranking::Declined { ranked, .. } => ranked,
        }
    }
}

/// Rank cells at floor `beta`, returning a verdict.
///
/// `lower_is_better` because most quantities ranked here are costs — journey time, price,
/// exposure. Pass `false` for a quantity where more is better, such as an offered price.
///
/// # ⚠️ Unknowns sort last and cannot lead
///
/// This is subtle enough to be worth the paragraph. An unknown cell has `lo == -infinity`, so
/// sorting naively on `lo` puts it **first** and hands it the leader slot on the strength of
/// knowing nothing. Worse, nothing can be separated from an infinite band, so a single unknown
/// would drag every other candidate into the contender set and force a decline — one unquoted
/// commodity suppressing a ranking that was otherwise decided.
///
/// ⭐ That is not honest decline; it is a bug wearing decline's clothes. Honest decline means
/// "the cells I have overlap". An unknown says "I have no cell", which is a reason to leave it
/// out of the comparison, not to abandon the comparison. So it stays in `ranked` — it is a live
/// hypothesis, and it dies by widening rather than by being evicted — but it sits at the back
/// and takes no part in deciding the leader.
///
/// ⚠️ The one case where an all-unknown field declines is when *nothing* was measured, and there
/// the decline is the true answer rather than a suppressed one.
pub fn rank(cells: &[Cell], beta: f64, lower_is_better: bool) -> Ranking {
    if cells.is_empty() {
        return Ranking::Declined {
            contenders: Vec::new(),
            ranked: Vec::new(),
        };
    }

    let mut known: Vec<usize> = (0..cells.len()).filter(|&i| !cells[i].is_unknown()).collect();
    let unknown: Vec<usize> = (0..cells.len()).filter(|&i| cells[i].is_unknown()).collect();

    let key = |i: usize| -> f64 {
        if lower_is_better {
            cells[i].lo
        } else {
            -cells[i].hi
        }
    };
    known.sort_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranked = known.clone();
    ranked.extend(unknown);

    if known.is_empty() {
        return Ranking::Declined {
            contenders: ranked.clone(),
            ranked,
        };
    }

    let best = cells[known[0]];
    let contenders: Vec<usize> = known
        .iter()
        .copied()
        .filter(|&i| !best.separated(&cells[i], beta))
        .collect();

    if contenders.len() == 1 {
        Ranking::Convergent {
            leader: known[0],
            ranked,
        }
    } else {
        Ranking::Declined { contenders, ranked }
    }
}

// ---------------------------------------------------------------------------
// Ageing
// ---------------------------------------------------------------------------

/// The reference interval against which staleness is measured, in milliseconds.
///
/// ⚠️ **Authored, and the one tuning constant in this module.** Ninety seconds is a transit
/// block headway in the source framework, and it is kept here because the exchange's own natural
/// interval is the same order: a market quote, a weather reading and a position fix are all
/// things that mean something different a few minutes later and nothing at all a few hours later.
///
/// What matters is not the value but the **shape**: widening is monotone in age, so evidence
/// eventually stops being separated from the field no matter what this is set to. Changing it
/// changes how fast that happens, never whether it happens.
pub const HEADWAY_MS: f64 = 90_000.0;

/// The uncertainty, in milliseconds, attributed to a timestamp with no quality signal at all.
///
/// ⚠️ Deliberately pessimistic. With no information about how a moment was measured, the honest
/// cell is wide — the same argument as [`crate::fusion::UNINFORMED_SIGMA_M`], one dimension down.
pub const DEFAULT_CLOCK_FLOOR_MS: f64 = 30_000.0;

/// How much to widen a cell whose evidence is `age_ms` old, at `rate` per headway.
///
/// ⭐ Linear, and deliberately so. This is the "sufficient, not perfect" principle: the property
/// that makes death-by-widening work is *monotone increase*, and a linear function has it. A
/// tuned exponential decay would add parameters that nobody could justify from a measurement,
/// which is the kind of invisible policy this codebase refuses elsewhere (see the ranking
/// argument in this crate's ranking exclusion).
///
/// Returns zero for a non-positive or non-finite age: evidence from the future is a clock
/// problem, and widening on it would let a bad clock erase good evidence.
pub fn staleness_widening(age_ms: f64, rate_per_headway: f64) -> f64 {
    if !(age_ms > 0.0) || !age_ms.is_finite() {
        return 0.0;
    }
    (age_ms / HEADWAY_MS) * rate_per_headway
}

/// Is this evidence too old to be worth re-using?
///
/// A convenience over [`staleness_widening`] for callers deciding whether to re-probe rather
/// than how much to widen.
pub fn is_stale(age_ms: f64, max_headways: f64) -> bool {
    age_ms > max_headways * HEADWAY_MS
}

/// A timestamp and the floor on it.
///
/// ⚠️ A moment is a cell too. A phone does not hold an atomic clock, it holds a disciplined
/// estimate, and the honest representation of "now" carries how well "now" is known.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Stamp {
    /// Epoch milliseconds.
    pub t_ms: f64,
    /// Half-width on `t_ms`.
    pub floor_ms: f64,
}

impl Stamp {
    pub fn new(t_ms: f64, floor_ms: f64) -> Stamp {
        Stamp {
            t_ms,
            floor_ms: floor_ms.max(0.0),
        }
    }

    /// A stamp with no quality signal.
    pub fn coarse(t_ms: f64) -> Stamp {
        Stamp::new(t_ms, DEFAULT_CLOCK_FLOOR_MS)
    }

    /// The age of this stamp as of `now`, together with the combined floor.
    ///
    /// ⚠️ Both stamps carry floors, so the age is itself uncertain, and the caller gets both
    /// numbers rather than a single one that hides the second.
    pub fn age_as_of(&self, now: Stamp) -> (f64, f64) {
        (now.t_ms - self.t_ms, self.floor_ms + now.floor_ms)
    }

    /// This moment as a cell.
    pub fn as_cell(&self) -> Cell {
        Cell {
            lo: self.t_ms - self.floor_ms,
            hi: self.t_ms + self.floor_ms,
            unit: CellUnit::Millis,
        }
    }
}

/// The floor implied by disagreement between independent clock sources.
///
/// ⭐ **Measured, not chosen**, and that is why this function exists rather than a constant.
/// Several independent constellations each estimate the receiver's clock bias; their spread is
/// directly observable per device and per moment. Clear sky gives a tight spread and a narrow
/// cell; a concourse gives a wide spread and a wide cell. No heuristic and no tuning.
///
/// ⚠️ Falls back to [`DEFAULT_CLOCK_FLOOR_MS`] with fewer than two sources, because disagreement
/// is not defined for one. That fallback is the honest answer, not a failure path.
pub fn clock_floor_from_sources(offsets_ms: &[f64]) -> f64 {
    let finite: Vec<f64> = offsets_ms.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 2 {
        return DEFAULT_CLOCK_FLOOR_MS;
    }
    let lo = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ((hi - lo) / 2.0).max(MIN_HALF_WIDTH)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(lo: f64, hi: f64) -> Cell {
        Cell::new(lo, hi, CellUnit::Millis).unwrap()
    }

    #[test]
    fn a_zero_floor_is_refused() {
        // ⭐ The Price-Cell claim, as a compile-and-run guarantee: no caller can construct a
        // point value through this module, however much they want one.
        assert_eq!(
            Cell::from_floor(10.0, 0.0, CellUnit::Money),
            Err(CellError::ZeroFloor)
        );
        assert_eq!(
            Cell::from_floor(10.0, -1.0, CellUnit::Money),
            Err(CellError::ZeroFloor)
        );
        assert!(Cell::from_floor(10.0, 0.5, CellUnit::Money).is_ok());
    }

    #[test]
    fn a_nan_bound_is_not_an_interval() {
        // ⚠️ The reason `new` is written `!(hi >= lo)`: NaN compares false to everything, so the
        // negated form catches it and the direct form would not.
        assert_eq!(
            Cell::new(f64::NAN, 1.0, CellUnit::Money),
            Err(CellError::NotAnInterval)
        );
        assert_eq!(
            Cell::new(0.0, f64::NAN, CellUnit::Money),
            Err(CellError::NotAnInterval)
        );
        assert_eq!(Cell::new(2.0, 1.0, CellUnit::Money), Err(CellError::NotAnInterval));
    }

    #[test]
    fn composing_accumulates_uncertainty_upward() {
        // ⭐ Nobody wrote error propagation; interval addition did it.
        let legs = [ms(10.0, 12.0), ms(20.0, 25.0), ms(5.0, 6.0)];
        let total = compose(&legs).unwrap();
        assert_eq!(total.lo, 35.0);
        assert_eq!(total.hi, 43.0);
        // The parent's band is wider than any child's.
        for leg in &legs {
            assert!(total.width() > leg.width());
        }
    }

    #[test]
    fn one_unknown_part_makes_the_whole_unknown() {
        let legs = [ms(10.0, 12.0), Cell::unknown(CellUnit::Millis)];
        assert!(compose(&legs).unwrap().is_unknown());
        assert!(compose_worst(&legs).unwrap().is_unknown());
    }

    #[test]
    fn composing_across_units_is_an_error_not_a_wide_result() {
        let mixed = [
            Cell::new(1.0, 2.0, CellUnit::Millis).unwrap(),
            Cell::new(1.0, 2.0, CellUnit::Money).unwrap(),
        ];
        assert!(matches!(
            compose(&mixed),
            Err(CellError::UnitMismatch { .. })
        ));
    }

    #[test]
    fn worst_case_takes_the_dominating_band_whole() {
        // ⚠️ The weather argument: the worst leg, keeping its band rather than its `hi`.
        let legs = [ms(1.0, 3.0), ms(2.0, 9.0), ms(0.0, 4.0)];
        let worst = compose_worst(&legs).unwrap();
        assert_eq!(worst.lo, 2.0);
        assert_eq!(worst.hi, 9.0);
        // Not the sum — which would have been [3, 16].
        assert_ne!(worst.hi, 16.0);
    }

    #[test]
    fn intersection_narrows_and_contradiction_is_reported() {
        let a = ms(10.0, 20.0);
        let b = ms(15.0, 30.0);
        let both = a.intersect(&b).unwrap().unwrap();
        assert_eq!((both.lo, both.hi), (15.0, 20.0));
        assert!(both.width() < a.width());

        // ⭐ Disjoint is `Ok(None)` — information, not an error, and never an average.
        let c = ms(100.0, 200.0);
        assert!(a.intersect(&c).unwrap().is_none());
    }

    #[test]
    fn separation_is_on_the_gap_not_the_midpoint() {
        // Midpoints differ by 10, but the bands touch: not separable.
        let a = ms(0.0, 10.0);
        let b = ms(10.0, 20.0);
        assert!(a.mid().unwrap() != b.mid().unwrap());
        assert!(!a.separated(&b, 0.0));

        let c = ms(30.0, 40.0);
        assert!(a.separated(&c, 5.0));
        assert!(!a.separated(&c, 25.0)); // gap is 20; a bigger floor swallows it
    }

    #[test]
    fn nothing_is_separated_from_an_unknown() {
        let u = Cell::unknown(CellUnit::Millis);
        assert!(!ms(0.0, 1.0).separated(&u, 0.0));
        assert!(!u.separated(&u, 0.0));
    }

    #[test]
    fn an_unknown_cannot_lead_a_ranking() {
        // ⭐ The bug this guards: `lo == -inf` sorts to the front, so a naive ranking would hand
        // the leader slot to the one candidate nothing is known about.
        let cells = [
            Cell::unknown(CellUnit::Money),
            Cell::new(10.0, 11.0, CellUnit::Money).unwrap(),
            Cell::new(50.0, 51.0, CellUnit::Money).unwrap(),
        ];
        let r = rank(&cells, 1.0, true);
        assert_eq!(r.leader(), Some(1));
        // It is still present — a live hypothesis, not evicted — but at the back.
        assert_eq!(r.ranked(), &[1, 2, 0]);
    }

    #[test]
    fn an_unknown_does_not_suppress_a_decided_ranking() {
        // ⚠️ The second half of the same bug: nothing is separable from an infinite band, so an
        // unknown left in the comparison would drag every candidate into the contender set.
        let cells = [
            Cell::new(10.0, 11.0, CellUnit::Money).unwrap(),
            Cell::new(90.0, 91.0, CellUnit::Money).unwrap(),
            Cell::unknown(CellUnit::Money),
        ];
        assert!(matches!(rank(&cells, 1.0, true), Ranking::Convergent { leader: 0, .. }));
    }

    #[test]
    fn overlapping_leaders_decline_and_name_their_contenders() {
        let cells = [
            Cell::new(10.0, 20.0, CellUnit::Money).unwrap(),
            Cell::new(15.0, 25.0, CellUnit::Money).unwrap(),
            Cell::new(900.0, 910.0, CellUnit::Money).unwrap(),
        ];
        match rank(&cells, 1.0, true) {
            Ranking::Declined { contenders, .. } => {
                // ⭐ The decline carries its classes: the caller learns 0 and 1 are the pair that
                // a further observation would have to separate, and that 2 is out of it.
                assert_eq!(contenders, vec![0, 1]);
            }
            other => panic!("expected decline, got {other:?}"),
        }
    }

    #[test]
    fn an_all_unknown_field_declines_honestly() {
        let cells = [Cell::unknown(CellUnit::Money), Cell::unknown(CellUnit::Money)];
        assert!(matches!(rank(&cells, 1.0, true), Ranking::Declined { .. }));
    }

    #[test]
    fn ranking_can_prefer_the_higher_band() {
        // An offered price: more is better.
        let cells = [
            Cell::new(10.0, 11.0, CellUnit::Money).unwrap(),
            Cell::new(90.0, 91.0, CellUnit::Money).unwrap(),
        ];
        assert_eq!(rank(&cells, 1.0, false).leader(), Some(1));
    }

    #[test]
    fn widening_is_monotone_in_age() {
        // ⭐ The only property death-by-widening needs.
        let a = staleness_widening(HEADWAY_MS, 1.0);
        let b = staleness_widening(HEADWAY_MS * 4.0, 1.0);
        assert!(b > a);
        assert_eq!(a, 1.0);
        // Evidence from the future widens nothing.
        assert_eq!(staleness_widening(-1000.0, 1.0), 0.0);
        assert_eq!(staleness_widening(f64::NAN, 1.0), 0.0);
    }

    #[test]
    fn stale_evidence_stops_being_separated_from_the_field() {
        // ⭐ The whole point of ageing, asserted end to end: two claims that were distinguishable
        // when fresh become indistinguishable once one of them is old. Nothing evicted it.
        let fresh = ms(0.0, 10.0);
        let other = ms(40.0, 50.0);
        assert!(fresh.separated(&other, 5.0));

        let aged = fresh.widen(staleness_widening(HEADWAY_MS * 40.0, 1.0));
        assert!(!aged.separated(&other, 5.0));
    }

    #[test]
    fn a_clock_floor_is_measured_from_disagreement() {
        // Tight agreement, narrow cell.
        assert!(clock_floor_from_sources(&[10.0, 12.0, 11.0]) < 2.0);
        // Wide disagreement, wide cell.
        assert!(clock_floor_from_sources(&[0.0, 10_000.0]) > 4_000.0);
        // ⚠️ One source cannot disagree with itself: the pessimistic default, not a zero.
        assert_eq!(clock_floor_from_sources(&[10.0]), DEFAULT_CLOCK_FLOOR_MS);
        assert_eq!(clock_floor_from_sources(&[]), DEFAULT_CLOCK_FLOOR_MS);
    }

    #[test]
    fn a_stamp_is_itself_a_cell() {
        let s = Stamp::new(1_000.0, 50.0);
        let c = s.as_cell();
        assert_eq!((c.lo, c.hi), (950.0, 1_050.0));
        assert_eq!(c.unit, CellUnit::Millis);

        let (age, floor) = s.age_as_of(Stamp::new(4_000.0, 20.0));
        assert_eq!(age, 3_000.0);
        // ⚠️ Both floors, because the age is uncertain at both ends.
        assert_eq!(floor, 70.0);
    }

    #[test]
    fn an_unknown_cell_has_no_midpoint() {
        assert!(Cell::unknown(CellUnit::Money).mid().is_none());
        assert!(Cell::unknown(CellUnit::Money).widen(100.0).is_unknown());
        assert_eq!(ms(0.0, 10.0).mid(), Some(5.0));
    }

    #[test]
    fn empty_input_is_an_error_not_a_zero() {
        // ⚠️ `compose(&[])` returning a zero-width cell at the origin would be a fabricated
        // quantity with no source at all.
        assert_eq!(compose(&[]), Err(CellError::Empty));
        assert_eq!(compose_worst(&[]), Err(CellError::Empty));
    }
}
