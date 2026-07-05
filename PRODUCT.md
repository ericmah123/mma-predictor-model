# Product

## Register

product

## Users

Two overlapping audiences on the same page: casual MMA fans settling a
"who wins" debate before a card or with friends, and more analytical
users (bettors, stat-heads) who want to see the model's reasoning before
trusting the number. The interaction is a single task — pick fighter A,
pick fighter B, get a probability — so the design must serve a fast fun
answer up front while keeping the detailed stat comparison one glance
away for anyone who wants to dig in.

## Product Purpose

Predict UFC fight outcomes head-to-head from a gradient-boosted model
trained on real, chronologically-replayed fight history. Success is a
user picking two fighters in seconds (via search/autocomplete), getting
a clear win-probability read, and being able to see *why* (the
per-stat differential table) without feeling like they're reading a
data dashboard.

## Brand Personality

Combat-sport intensity: bold, high-contrast, a little aggressive — the
energy of fight-night broadcast graphics, not a SaaS product. Confident
about the numbers (this is a real model, not a gimmick) without being
dry or clinical.

## Anti-references

- Generic SaaS dashboard: purple/indigo gradient hero-metric clichés,
  soft rounded cards, glassmorphism-for-its-own-sake. The current UI
  leans this way (indigo/pink gradient theme) and should move toward
  something with more combat-sport identity.
- Gambling/sportsbook aesthetics: neon odds-board styling, casino
  urgency cues, countdown pressure tactics. This is a prediction tool
  grounded in a real model, not a betting product.

## Design Principles

1. **Fast fun front, rigor on demand** — the headline verdict and
   probability should be legible instantly; the stat-by-stat breakdown
   is available immediately below for anyone who wants the "why."
2. **Earn the intensity, don't fake it** — bold/aggressive visual energy
   should come from typography, contrast, and motion, not from
   decorative gradients or glow-for-glow's-sake.
3. **Respect the model's honesty** — the held-out accuracy and
   volatility disclaimer are part of the product's credibility; never
   let visual design bury or undersell them.
4. **Symmetry reflects the math** — the two-fighter, side-by-side
   structure (corner A / corner B) mirrors the model's differential,
   swap-invariant design and should stay visually balanced.

## Accessibility & Inclusion

Standard WCAG AA: body text ≥4.5:1 contrast, large text ≥3:1, visible
keyboard focus states on all interactive elements (search inputs,
suggestion lists, buttons), and a `prefers-reduced-motion` alternative
for any probability-bar or reveal animation.
