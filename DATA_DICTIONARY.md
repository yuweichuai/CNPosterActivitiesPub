# Processed data dictionary

## Keys and observation units

| Field | Meaning |
| --- | --- |
| `tweet_id` | Pseudonymous focal event key, including treated events and control pseudo-events. |
| `user_id` | Pseudonymous author account key, consistent across all files. |
| `treated` | Original event-level treatment indicator, 1 for displayed-note events and 0 for controls. Some heterogeneity notebooks explicitly replace this indicator for their stated comparison, as in the original code. |
| `days_from_display` | Relative day of activity around actual or assigned pseudo-correction time. Count panels contain −28 through −1 and +1 through +28; the short daily analyses select −4 through −1 and +1 through +4. |
| `weeks_from_display` | Relative week, −4 through −1 and +1 through +4. |
| `account_cluster` | Created at runtime from `user_id`; not a separate real identifier. |

IDs are strings. Preserve them as strings in external software. Repeated account IDs across events and treatment groups are intentional. Joining event-period inputs to source characteristics uses `tweet_id`; inference clusters on the pseudonymous account identifier where specified.

## Event characteristics

`source_post_features.parquet` contains the processed event characteristics used in the sample description. `source_post_features_processed.parquet` additionally contains prior-NMR history flags/counts for the main analyses. Both retain their original row order. Original post text and unused features are removed.

| Field or family | Meaning |
| --- | --- |
| `note_score` | Stored helpfulness score of the focal displayed note; used in the note-score comparison. |
| `total_corrections` | Total observed corrections associated with the account in the stored history; retained for descriptive summaries and the existing eligibility logic. |
| `current_correction` | Stored correction order of the focal event. |
| `prior_corrections` | Stored count of prior corrections used to define correction-history strata. |
| `days2previousNote`, `days2nextNote` | Stored intervals to neighboring correction events, in days. Missing values are preserved. |
| `control_has_correction_{h}d` | Existing control-window contamination flag for h = 4, 8, 28, 56. `get_source_corrections()` applies the flag for the requested window. |
| `user_follower_count`, `user_following_count` | Stored profile counts used as account covariates. |
| `user_is_verified` | Stored verified-status indicator. |
| `user_account_age_days` | Stored account age in days. |
| `partisan_score`, `misinfo_exposure_score` | Stored account-level political-leaning and misinformation-exposure measures. |
| `days_from_rollout` | Relative time from rollout, in days; used as a covariate and to determine eligibility for prior-history horizons. |
| `UserPostFrequency` | Stored pre-correction posting-frequency measure. |
| `UserPostPositive`, `UserPostNegative`, `UserPostToxicity` | Stored pre-correction content-characteristic averages. |
| `UserPostMisleadingness` | Stored pre-correction GPT-estimated misleadingness average. Subgroup cutoffs and transformations are calculated in the notebooks. |
| `UserPostConfidence`, `UserPostPolitics` | Stored pre-correction opinion-confidence and political-content measures. |
| `reply_label` | In the descriptive source table only: 1 when the original focal-post text began with `@`, following the original notebook's rule. The text itself is excluded. |
| `event_creation_order` | In the main source table only: dense rank of focal-post creation time. Reproduces chronological sorting in 3_1 without releasing exact timestamps. |
| `nmr_eligible_{h}d` | Eligibility for the h-day prior-NMR window, using the original `days_from_rollout > h` rule; h = 4, 28, 60, 90, 180. |
| `nmr_count_{h}d` | Stored count of prior annotated posts represented by retained NMR notes in the lookback window. The original preparation selects potentially misleading classifications and retains the earliest NMR note per post before counting. |
| `nmr_high_score_count_{h}d` | Corresponding count restricted to retained notes with `note_score >= 0.30`. |

These covariates are distributed on their stored scales. The notebooks retain their own log transformations, standardization, median/quartile splits, and sample-specific exclusions. Frozen content-matched membership contains IDs only, so downstream analyses reload the original covariate scales instead of accidentally using the standardized matching covariates.

## Posting panels and processed sample membership

| Dataset | Unit and fields |
| --- | --- |
| `ts_posts_daily.parquet.gzip` | Event-day: `tweet_id`, `days_from_display`, `tweet_count`. |
| `ts_posts_weekly.parquet.gzip` | Event-week: `tweet_id`, `weeks_from_display`, `tweet_count`. |
| `deleted_posts_daily.parquet` | Deleted focal-event-day: event key, account key, relative day, `tweet_count`. |
| `deleted_posts_weekly.parquet` | Deleted focal-event-week: event key, account key, relative week, `tweet_count`. |
| `source_corrections_content_matched.parquet` | Frozen matched event membership for the content analyses; one `tweet_id` per event. |
| `source_corrections_note_score.parquet` | Processed event membership for the CoreModel control-score robustness check, with `note_score_control`. Controls were selected at a score of at least 0.35 in the original preparation. |

`tweet_count` is the original API-based count outcome. Existing zero counts, eligibility restrictions, and missing histories are not changed by packaging. The full supplied panels precede the analysis-specific selection performed in each notebook. The deleted-post panels contain 78 events with valid histories, each with 56 daily and eight weekly observations.

## Descriptive account tables

- `original_post_counts_by_account.parquet`: one row per account in the retrieved-post dataset; `original_post_count` is the number of retrieved original-post rows. Notebook 1_1 applies the original treatment/control account-membership joins before summing. Accounts shared by the groups contribute to both group totals, while the overall total uses the account union.
- `fixed_horizon_accounts.parquet`: one row per account in the descriptive correction sample. `available_followup_days` is the original duration from first observed correction to the June 11, 2024 history cutoff. `first_correction_quarter` is the coarse entry cohort. `missing_history` identifies accounts without a first correction in the cutoff-restricted history. For h = 4, 28, 60, 90, 180, `eligible_{h}d` indicates complete follow-up and `repeated_within_{h}d` indicates at least two corrections within that horizon. The original inclusive horizon endpoint is retained; ineligible recurrence values remain missing.
- `extended_correction_accounts.parquet`: account totals for the 60-day-eligible comparison. `total_corrections` is the original total, `total_corrections_extended` is the total after the original historical-record merge, and `original_group` is `One correction` or `Repeated corrections` according to recurrence within 60 days. Historical matching was completed before removing handles; exact source joins are recorded in the manifest.

## Content-count tables

Every table under `data/content_counts/` contains a focal event key, relative day/week, and two columns named `tweet_count_{variable}_yes` and `tweet_count_{variable}_no`. They count classified original posts with a score greater than or equal to, or below, the threshold. They reproduce the original helper's count of nonmissing original-post IDs after its score joins. Neither original-post IDs nor post text is needed for the model stage.

| Variable | Threshold for `yes` | Time unit |
| --- | --- | --- |
| `misleadingness`, overall content analysis | GPT score ≥ 0.86 | Daily, short window |
| `misleadingness`, high-baseline and prior-NMR analyses | GPT score ≥ 0.86 | Weekly, four-week window |
| `positive` | Score ≥ 0.5 | Weekly |
| `negative` | Score ≥ 0.5 | Weekly |
| `toxicity` | Score ≥ 0.5 | Weekly |
| `confidence` | Score ≥ 6.5 | Weekly |
| `politics_is` | Indicator ≥ 1 | Weekly |

The two combined misleadingness inputs reproduce the original post-level selections: `politics_is == 1 or toxicity >= 0.5`, and `politics_is == 0 and toxicity < 0.5`. These are **political or highly toxic** versus **non-political and low-toxicity** categories. The original comparison is preserved; no new between-category contrast is introduced.

Category-count files preserve the missing side of an outer join. The notebooks subsequently merge these counts into the API posting panel and fill missing category counts with zero, as before. The package does not equate a classifier failure with a validated negative label. Content shares use the original `tweet_count` denominator and are estimated only under the original positive-count and valid-ratio restrictions.

`data/content_counts/manifest.json` maps source aliases in the notebooks to these ten processed tables. It records the score threshold and time unit for every table. For confidence and short-window misleadingness, the original score joins and deduplication were completed before aggregation.

## Reply scores and engagement

`fact_replies.parquet` contains one row per reply used in 3_5, with `tweet_id` referring to its focal correction event, `moral_outrage_probability`, and `displayed` (`Before` or `After`). The indicator is calculated from the original reply and note timestamps. The reply-level score distribution is retained for the mean/contrast calculation; original reply IDs and authors are excluded.

`post_engagement_weekly.parquet` contains `tweet_id`, `weeks_from_display`, `user_tweet_retweets`, and `user_tweet_replies`. The last two are rounded event-week **mean engagement per original post**, using the original Python `round` operation, not weekly totals. Notebook 6 applies the original matched-sample restriction before fitting its models.
