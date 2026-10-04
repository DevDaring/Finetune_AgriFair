"""Comparisons removed after the main runs, with the reason. Every analysis drops them.

ext-120 (added 4 Oct 2026, found while preparing the human spot-check): its census cell
(T14-16 ST/femaleShare/area/MediumvsSmall) gives women's share of operated area within medium and
within small holdings. The generator rendered it with the size-class layout, so the question asks
about "Scheduled Tribe holders with femaleshare holdings" and compares medium with small operated
area, which the source does not support. The size-class check passed it because it looks for the
stratum word, and the generator had copied that raw field value into the question.
"""
EXCLUDED_COMPARISONS = {"ext-120"}
