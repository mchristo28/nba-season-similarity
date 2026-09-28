# Position peer release — model 3.2

Fetched September 28, 2026: 689 team-season rosters covering 2003–04 through 2025–26.
The NBA CommonTeamRoster endpoint was requested with each season and team explicitly;
returned season and team identifiers were checked. Labels are joined by player ID
and season, preserving original team labels and hybrid memberships.

Coverage is 10,507 / 11,501 player-seasons (91.4%) and 7,000 / 7,103 qualifying
rotation-player seasons (98.5%). The latest season has 525 / 582 listed positions,
including 343 / 350 rotation players. Missing current rotation-player labels include
Cameron Payne, Lonzo Ball, Cole Anthony, Jeremiah Robinson-Earl, Cam Thomas,
Jaden Ivey, and Vince Williams Jr. These players remain searchable with All players.
No current-position or height-based fallback is applied to historical seasons.

Position peers is the default, with an All players fallback for unknown subjects.
Users can add groups to expand the reference or choose All players explicitly.
Roster snapshots do not constitute
an exhaustive record of everyone who played during a season, or actual minutes by
position. F alone does not reliably separate wings from power forwards.

The Guard reference contains 2,130 qualifying seasons in Tracking mode. For Keyonte
George and Austin Reaves (2025–26), rim attempt share changes from a 0.445-SD gap
(Close) against all players to 0.699 SD (Noticeable) against guards. Midrange changes
from 0.890 to 0.987 SD and remains Noticeable. Their overall score changes from
88.27 to 84.26. See `position-peer-audit.json` for the recorded measurements.

All standardized matrices were compared with the original, unenriched feature file
for Historical, Tracking, and Production: they are exactly unchanged in All players
mode. Tests also cover hybrid group switches, missing positions, cohort isolation,
search-filter independence, score/explanation agreement, and saved-model restoration.

Reference: https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/commonteamroster.md
