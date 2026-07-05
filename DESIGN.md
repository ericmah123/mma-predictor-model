---
name: MMA Fight Predictor
description: Head-to-head UFC win-probability tool, styled like the fight itself — red corner vs. blue corner on a dark canvas.
colors:
  corner-red: "oklch(0.58 0.20 25)"
  corner-blue: "oklch(0.52 0.18 250)"
  champion-gold: "oklch(0.74 0.15 85)"
  canvas-black: "oklch(0.14 0 0)"
  concrete-surface: "oklch(0.19 0.006 30)"
  concrete-surface-raised: "oklch(0.24 0.007 30)"
  chalk-ink: "oklch(0.97 0.003 90)"
  chalk-muted: "oklch(0.70 0.01 90)"
  rope-border: "oklch(0.30 0.01 30)"
typography:
  display:
    fontFamily: "Oswald, 'Arial Narrow', sans-serif"
    fontSize: "clamp(2.4rem, 5vw, 3.6rem)"
    fontWeight: 600
    lineHeight: 1.05
    letterSpacing: "-0.01em"
  headline:
    fontFamily: "Oswald, 'Arial Narrow', sans-serif"
    fontSize: "1.5rem"
    fontWeight: 500
    lineHeight: 1.2
    letterSpacing: "-0.005em"
  body:
    fontFamily: "Inter, 'Segoe UI', sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "Inter, 'Segoe UI', sans-serif"
    fontSize: "0.85rem"
    fontWeight: 600
    letterSpacing: "0.02em"
  stat-mono:
    fontFamily: "'IBM Plex Mono', 'Consolas', monospace"
    fontSize: "0.95rem"
    fontWeight: 500
rounded:
  sm: "4px"
  md: "6px"
  none: "0px"
spacing:
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
  2xl: "48px"
components:
  button-primary:
    backgroundColor: "{colors.corner-red}"
    textColor: "{colors.chalk-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "16px 44px"
  button-primary-hover:
    backgroundColor: "oklch(0.52 0.21 25)"
    textColor: "{colors.chalk-ink}"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.chalk-muted}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "16px 28px"
  badge-corner-a:
    backgroundColor: "{colors.corner-red}"
    textColor: "{colors.chalk-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "4px 12px"
  badge-corner-b:
    backgroundColor: "{colors.corner-blue}"
    textColor: "{colors.chalk-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "4px 12px"
  input-search:
    backgroundColor: "{colors.canvas-black}"
    textColor: "{colors.chalk-ink}"
    rounded: "{rounded.md}"
    padding: "12px 14px"
  card-fighter:
    backgroundColor: "{colors.concrete-surface}"
    textColor: "{colors.chalk-ink}"
    rounded: "{rounded.md}"
    padding: "24px"
---

# Design System: MMA Fight Predictor

## 1. Overview

**Creative North Star: "The Fight Corner"**

The system is built around the one piece of MMA visual language everyone
already recognizes without being told: red corner versus blue corner. Corner
A is always ember red, Corner B is always cobalt blue — not as a decorative
accent choice but as the literal convention the sport uses on every fight
card. The canvas underneath is near-black concrete, lit like an arena at
night, so the two corner colors are the only saturated things in the room.
A third color, champion gold, is held in reserve — it only appears to mark
the stat or fighter the model favors, so it reads as a result, not a
decoration.

This directly rejects PRODUCT.md's anti-references: no indigo/pink SaaS
gradient, no soft glassmorphism, no gambling-site neon odds-board energy.
The intensity here comes from contrast and structure (a hard-edged red vs.
blue matchup on black), not from glow-for-glow's-sake or decorative
gradients.

**Key Characteristics:**
- Two-color combat, not a palette: red corner, blue corner, nothing else
  saturated except the gold "favored" marker.
- Near-black canvas with tonal layering, not drop shadows, for depth.
- Condensed, bold display type (broadcast-graphic energy) paired with a
  clean data-friendly body face and a monospace face for numbers.
- Cut-corner primary actions instead of pill buttons — a ticket-stub cue,
  not a SaaS rounded-everything default.

## 2. Colors

Two saturated corner colors carry the identity; everything else is a
near-black tonal scale that stays out of their way.

### Primary
- **Corner Red** (`oklch(0.58 0.20 25)`): Corner A's color everywhere it
  appears — badge, input focus ring, primary button, prob-bar fill,
  fighter-card accent border. Always paired with white/near-white text
  (`chalk-ink`), never dark text, per the Helmholtz-Kohlrausch rule for
  saturated mid-lightness fills.

### Secondary
- **Corner Blue** (`oklch(0.52 0.18 250)`): Corner B's color, used
  identically and symmetrically to Corner Red — same components, mirrored
  side. The two must never blend, gradient into each other, or share a
  component; the whole point is that they stay two distinct corners.

### Tertiary
- **Champion Gold** (`oklch(0.74 0.15 85)`): Reserved exclusively for
  marking a result — the favored stat in the comparison table, the winning
  corner's glow after a prediction. Never used as a resting decorative
  accent; if gold is visible, it means "this one is winning."

### Neutral
- **Canvas Black** (`oklch(0.14 0 0)`): Page background. Pure neutral, no
  hue tint — the arena-at-night backdrop the two corner colors sit on.
- **Concrete Surface** (`oklch(0.19 0.006 30)`): Fighter cards, results
  panel, dropdown backgrounds. One step up from canvas-black.
- **Concrete Surface Raised** (`oklch(0.24 0.007 30)`): Hover/active state
  for surfaces, and the resting tone for the winning fighter-card after a
  prediction is rendered.
- **Chalk Ink** (`oklch(0.97 0.003 90)`): Primary text. ~9:1+ contrast
  against canvas-black.
- **Chalk Muted** (`oklch(0.70 0.01 90)`): Secondary text (labels, helper
  copy, picked-fighter subtext). Still clears 4.5:1 against canvas-black.
- **Rope Border** (`oklch(0.30 0.01 30)`): Dividers, card borders, input
  borders at rest.

### Named Rules
**The Corner Rule.** Corner A is always ember red, Corner B is always
cobalt blue. Never swapped, never tinted toward each other, never rendered
as a gradient blend — the two-color split *is* the design, mirroring the
model's own symmetric, differential math.

**The Earned Gold Rule.** Champion gold only appears attached to an actual
result (a favored stat, a predicted winner). It is forbidden as ambient
decoration — no gold hairlines, no gold icons "for warmth."

## 3. Typography

**Display Font:** Oswald (with "Arial Narrow", sans-serif fallback)
**Body Font:** Inter (with "Segoe UI", sans-serif fallback)
**Label/Mono Font:** IBM Plex Mono (with "Consolas", monospace fallback), for numeric readouts

**Character:** Oswald is condensed and bold — the broadcast-lower-third
energy of a fight-night graphic. Inter carries the actual reading (body
copy, inputs, table stats) so density never feels shouted. IBM Plex Mono
marks every number that matters (win probabilities, records, stat values)
as data, not prose — a small but deliberate nod to a stat sheet.

### Hierarchy
- **Display** (600, `clamp(2.4rem, 5vw, 3.6rem)`, 1.05): The `<h1>` and the
  post-prediction verdict ("Jon Jones is favored to win"). Condensed and
  tight; `text-wrap: balance` required so it never breaks mid-phrase.
- **Headline** (500, 1.5rem, 1.2): Section titles ("How it works", results
  panel heading).
- **Title** (600, 1.1rem, 1.3): Card-level headings (hero-card `<h2>`).
- **Body** (400, 1rem, 1.55): Paragraphs, subtitle copy, disclaimer text.
  Cap prose at 65-75ch.
- **Label** (600, 0.85rem, tracking 0.02em): Badge text, button text,
  input labels. Uppercase only on badges (`Corner A` / `Corner B`), never
  as a decorative section eyebrow.
- **Stat-mono** (500, 0.95rem, monospace): Win probability percentages,
  fight records (12-3), and every cell in the comparison table.

### Named Rules
**The No-Eyebrow Rule.** No small-caps tracked label sits alone above a
section as decoration (the current `.eyebrow` pattern). If a short tag is
needed above the hero, it must attach to real information — a bout-style
tag ("Model v3 · Real UFC data") — never a content-free kicker repeated
above every section.

## 4. Elevation

Flat by default, depth conveyed through tonal layering
(`canvas-black` → `concrete-surface` → `concrete-surface-raised`), not
drop shadows. Glow is the one exception, and it is earned, not ambient: it
only appears as a direct response to state — hover/focus on interactive
elements, and around the winning fighter's card once a prediction resolves.

### Shadow Vocabulary
- **Ember Focus** (`box-shadow: 0 0 0 3px oklch(0.58 0.20 25 / 0.35)`): Focus
  ring on Corner A's input/button; swap hue to `oklch(0.52 0.18 250 / 0.35)`
  for Corner B's equivalents.
- **Winner Glow** (`box-shadow: 0 0 32px oklch(0.74 0.15 85 / 0.25)`): Applied
  once, to the favored fighter's card, after a prediction renders. Never at
  rest, never on both cards at once.

### Named Rules
**The Earned Glow Rule.** No ambient glow at rest, anywhere. Every glow in
this system is a direct answer to a state change (focus, hover, "this
fighter just won the prediction") — never decoration.

## 5. Components

### Buttons
- **Shape:** Sharp rectangle with one clipped corner via `clip-path`
  (`polygon(0 0, 100% 0, 100% calc(100% - 14px), calc(100% - 14px) 100%, 0 100%)`)
  on the primary action only — a ticket-stub cue, not a rounded pill.
- **Primary ("Predict fight"):** `corner-red` fill, `chalk-ink` text,
  Label typography, uppercase, 16px/44px padding. Hover: darken fill
  slightly + `translateY(-2px)`, no gradient.
- **Hover / Focus:** `translateY(-2px)` + Ember Focus ring on
  `:focus-visible`. No box-shadow at rest.
- **Secondary ("Reset"):** Transparent background, `rope-border` outline,
  `chalk-muted` text, plain rectangle (no clipped corner — that cue is
  reserved for the primary action only).

### Badges (Corner A / Corner B)
- **Style:** Filled rectangle, `rounded.sm` (4px), corner-red or
  corner-blue background, `chalk-ink` text, Label typography, uppercase.
- **State:** Static — badges don't have interactive states; they're
  identity markers, not controls.

### Cards / Containers (fighter-card, results panel)
- **Corner Style:** `rounded.md` (6px) — a tight radius, not the previous
  20px pill-adjacent curve.
- **Background:** `concrete-surface`.
- **Shadow Strategy:** None at rest. Winner Glow applies post-prediction
  to whichever fighter-card corresponds to the favored fighter.
- **Border:** 1px `rope-border`; Corner A's card border shifts to
  `corner-red` at 30% opacity, Corner B's to `corner-blue` at 30% opacity,
  so the split reads even before any input is typed.
- **Internal Padding:** `spacing.lg` (24px).

### Inputs / Fields
- **Style:** `canvas-black` background (one step darker than the card
  it sits in, so it reads as a "slot"), 1px `rope-border`, `rounded.md`.
- **Focus:** Ember Focus / matching corner-color ring, no color-changing
  border animation beyond the ring.
- **Suggestions dropdown:** `concrete-surface-raised` background, divided
  by 1px `rope-border` hairlines between rows instead of hover-only
  highlighting, so the list reads as a stat sheet, not a generic menu.

### Comparison Table
- **Header:** Corner A column in `corner-red`, Corner B column in
  `corner-blue`, center stat-name column in `chalk-muted` — Label
  typography, uppercase.
- **Values:** Stat-mono typography throughout.
- **Favored cell:** `champion-gold` text, no background fill (gold marks a
  result, it doesn't need a pill to do it).

### Probability Bar (signature component)
- Two-segment fill, not a single gradient: the Corner A share renders in
  `corner-red`, the remainder in `corner-blue` — literally the same split
  as the badges and card borders. This replaces the previous single
  gradient-fill-on-pink-track pattern with a direct visualization of the
  matchup instead of a decorative bar.

## 6. Do's and Don'ts

### Do:
- **Do** keep Corner A red and Corner B blue everywhere — badge, card
  border, input focus ring, probability bar segment. One consistent split,
  reinforced in every component.
- **Do** use tonal layering (`canvas-black` → `concrete-surface` →
  `concrete-surface-raised`) for depth; reserve glow for state changes
  only (hover, focus, the post-prediction winner reveal).
- **Do** run every number (probabilities, records, stat values) in
  Stat-mono so data reads as data.
- **Do** clip one corner on the primary button only; every other shape in
  the system stays a sharp rectangle or a tight 6px radius.

### Don't:
- **Don't** reintroduce the indigo/pink gradient theme, or any
  `linear-gradient` accent fill — PRODUCT.md names this directly as the
  generic-SaaS look to move away from.
- **Don't** style anything like a sportsbook: no neon odds-board colors,
  no countdown-timer urgency, no "+150" style odds formatting.
- **Don't** add glassmorphism or blur-for-its-own-sake; the concrete
  surfaces are opaque and flat.
- **Don't** use `background-clip: text` gradients anywhere, including the
  verdict headline — it stays solid `chalk-ink`.
- **Don't** put a tiny uppercase tracked eyebrow above a section purely as
  decoration (see The No-Eyebrow Rule) — the closest existing instance
  (the hero `.eyebrow`) should become a real bout-tag, not a bare kicker.
- **Don't** round buttons into full pills; the cut-corner primary button
  and 6px-radius everything-else replace the current 999px pill/20px card
  radius entirely.
