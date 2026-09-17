# Speaker Script — Healthcare Risk Intelligence Platform
### Target runtime: 10–12 minutes · 10 slides · practice out loud, don't read word-for-word

General pacing note: talk to the room, not the slide. The bracketed timings are cumulative — if you're more than ~30 seconds off pace at a checkpoint, tighten the next section rather than rushing the whole rest of the talk. Slide 8 (the model) is the technical center of gravity — it's fine for that one to run a little long if something else runs a little short. This version lands around 11:40, so there's a little room to slow down but not much — if you're running long in rehearsal, trim slide 6 or the honest-data caveat on slide 7 first, since Q2 and Q3 already got a thorough treatment on slide 5's dashboard and in the callout box.

---

## Slide 1 — Title
**[0:00 → 0:30]**

"Good [morning/afternoon] — I'm Nadiya, and this is my capstone project for the Ironhack Data Analytics bootcamp: a healthcare risk intelligence platform built around one core question — can we understand, and even predict, which patients are most likely to come back to the hospital.

Over the next ten minutes I'll walk through four business questions, how I answered each one with data, and a live tool I built on top of it."

*[Advance]*

---

## Slide 2 — The Problem
**[0:30 → 1:40]**

"Let's start with why this matters. Hospitals are increasingly measured — and financially penalized — on 30-day readmissions. A patient bouncing back within a month usually means something was missed the first time: the discharge plan, the medication, or the underlying risk just wasn't well understood.

I framed this project around three concrete questions: who is at risk, why do they keep coming back, and is what we're prescribing them actually safe. And a fourth question ties it all together — once we understand these patterns, can we predict risk *before* it happens, not just explain it after the fact."

*[Advance]*

---

## Slide 3 — Data & Approach
**[1:40 → 2:50]**

"To answer these, I combined two real data sources. The first is the MIMIC-IV Demo dataset — 100 de-identified patients and 275 hospital admissions, with diagnoses, admission type, length of stay, and readmission outcomes. The second is OpenFDA's adverse event API — a public feed of reported drug side effects that I pulled live, which gives an independent view of medication safety.

On the technical side, this project covers the full analytics pipeline: Python and pandas for cleaning and wrangling three separate data sources into one merged schema; SQL for querying it; exploratory data analysis and statistics to find the patterns; two machine learning models to test predictability; and both Streamlit and Tableau to deliver the results to two different audiences — one interactive, one for a polished executive readout.

One thing I want to flag up front: every number in this deck is recomputed directly from my current processed data, not eyeballed off a chart — so what you're seeing is verified."

*[Advance]*

---

## Slide 4 — Executive Overview
**[2:50 → 3:35]**

"So — the population at a glance. 100 patients, 275 admissions, and a 19.3% thirty-day readmission rate — that's about one in five admissions coming back within a month. Using a simple, explainable rule — three or more admissions, or any 30-day readmission — 35% of patients already qualify as high-risk before I build any model at all.

That's the baseline. Now let's break down who's actually driving that number."

*[Advance]*

---

## Slide 5 — Q1: Who's at Risk
**[3:35 → 5:05]**

"First question: which patient groups are highest risk. This slide is a straight screenshot of my actual Tableau dashboard — I wanted you to see the real, interactive version, not a recreation.

By age, the under-30 group shows the highest readmission rate at 30%, though that's only 10 admissions, so I'm treating it as directional, not a firm conclusion. The next-highest is actually the 45-to-59 group at 22.5%, followed closely by 60-to-74 at 21.3% — both on a much larger base, so those two are the more statistically meaningful findings.

By admission type, direct emergency admissions readmit at 33% — the highest of any category — with regular ER admissions close behind at 24%.

By insurance, Medicaid patients readmit at nearly 32% — more than double the rate of Medicare patients at 14.4%. That's a real access-and-continuity-of-care signal, not just noise.

And this same dashboard breaks it down by diagnosis too, which I'll come back to in a moment. If you want to explore any of this yourself, there's a QR code on the slide linking to the live, interactive version.

The takeaway: age, admission type, and insurance all matter, but none alone tells the whole story — which is exactly what the predictive model later tries to untangle."

*[Advance]*

---

## Slide 6 — Q2: Diagnoses
**[5:05 → 6:25]**

"Second question: which diagnoses are most associated with repeat admissions. I ranked the top ten diagnoses by volume, and rather than make you read bar heights off a chart, I just listed each one with its readmission rate stated directly — so there's nothing to interpret, it's right there.

The two diagnoses with the highest readmission rates — other encephalopathy and coronary artery disease with angina — both come in at 33%, though on a small base of three admissions each, so again, directional. Acute kidney failure is the more robust finding: nearly one in three of its seven admissions readmit.

One honesty note here: I excluded admissions whose diagnosis code fell outside my code-to-label lookup — labeled 'Other, not mapped' — so this ranking reflects only real, named diagnoses, not a catch-all bucket."

*[Advance]*

---

## Slide 7 — Q3: Medication Safety
**[6:25 → 7:45]**

"Third question: which medications carry the highest reported adverse events. This is where I want to lead with a caveat, not end with one: OpenFDA is a public feed, completely independent of the 100 patients in this project — it answers this question on its own, not linked patient-by-patient.

I've listed these two side by side rather than charting them, and I'm counting *distinct safety reports*, not raw rows — a report that mentions the same drug or reaction twice no longer gets counted twice. With that: LETAIRIS alone shows up in 61 of the 100 total reports — 61% of the entire feed. That's not a safety signal about the drug specifically, it's a data concentration — a handful of reporters filing repeatedly. And it's worth knowing that 122 of the 145 distinct medications in this feed have only a single report each, so anything beyond the top few names should be read as directional, not statistically robust. Dyspnoea, or shortness of breath, is the most commonly reported adverse reaction overall, showing up in 11 of the 100 reports.

There's a QR code here too, linking to the interactive Medication Safety dashboard."

*[Advance]*

> **Before you present — Tableau still needs a manual step (verified this session):** I checked both live Tableau Public dashboards directly against the current data, and neither is fully in sync yet:
> - **Medication Safety** — your local workbook file already has the correct fix for "Top Medications by Report Volume" (verified against the packaged extract: Zolpidem=1, Letairis=61, matching the numbers on this slide), but the **live published page still shows the old numbers (Zolpidem=13, Letairis=144)** — it just needs to be republished/overwritten to Tableau Public to catch up. Two more things worth fixing while you're in there: "Most Reported Adverse Reactions" and "Serious Outcome Breakdown" are still using raw row counts instead of distinct-report counts (the same bug, un-fixed on those two sheets), and the medication list is sorted alphabetically rather than by volume — right-click each measure and switch to Count Distinct of `safetyreportid`, then sort descending.
> - **Predictive Analysis** — the confusion matrix and feature-importance sheets are already correct and live. The **"Model Performance Comparison" sheet is still stale** (it shows Random Forest accuracy = 0.58, matching Logistic Regression, instead of the correct 0.623) — right-click the `model_comparison_long` data source → Refresh, then republish.
>
> Do this before demo day and both QR codes are safe to invite people to scan live.

---

## Slide 8 — Q4: Predictive Modeling
**[7:45 → 9:25]**

"And now the question that ties it together: can we actually predict this? I trained and compared two models — Logistic Regression and Random Forest — on an 80-20... sorry, a 75-25 train-test split, using only information known *at admission time*, specifically to avoid leaking future information into the prediction.

Random Forest technically has higher accuracy, 62% versus 58%. But look at recall — Logistic Regression catches 69% of true readmissions, versus just 31% for Random Forest. I chose Logistic Regression deliberately: in this use case, missing a patient who *will* be readmitted is far more costly than a false alarm, so I optimized for catching real risk over a cleaner-looking accuracy number.

On the right, you can see what actually drives the model's predictions: emergency-route admissions carry the strongest signal — direct emergency and emergency-room admissions raise predicted risk the most — followed by a patient's own history of prior admissions. Age, interestingly, *lowers* predicted risk here, and diagnosis count on its own barely moves the needle once prior admissions are already in the model.

I'll say this plainly: F1 score is 0.38, trained on only 275 admissions. This is a transparent, leakage-free proof of concept — not a clinically validated tool. But it demonstrates the full, correct methodology you'd scale up with more data. Again, the QR code here links to the live Predictive Analytics dashboard if you'd like to explore it further."

*[Advance]*

---

## Slide 9 — Live Demo: Streamlit
**[9:25 → 10:35]**

"Everything I just walked through is also available as a live, interactive tool — a five-page Streamlit app.

[**If presenting live:** switch to the running app now and click through Home → Predict Readmission Risk.]
[**If using the screenshots:** narrate them —] "On the home page, a user gets the same KPIs and risk split I just showed you, plus navigation into each business question. And on the prediction page, it's a simple input-output tool: a form for a patient's age, diagnosis count, prior admissions, and admission type as the input, and a live-scored 30-day readmission risk — Low, Medium, or High — as the output, with a gauge chart and the same model driving it."

*[Advance]*

---

## Slide 10 — Limitations & Thank You
**[10:35 → 11:40]**

"To close, I want to be upfront about scope. This is a small sample — 100 patients, 275 admissions — so findings are directional, not statistically bulletproof. The medication data isn't linked to these specific patients. And the model is a proof of concept, not a clinical decision tool.

The natural next steps: validate this on a larger, prospective patient cohort, link medication history directly to individual records instead of an independent feed, and track model performance over time as new admissions come in.

Thank you — feel free to scan the QR code to connect with me on LinkedIn, and I'd love to take any questions."

**[~11:40 total]**

---

## Anticipated Questions

**"Why Logistic Regression over Random Forest if accuracy is lower?"**
Because recall matters more than accuracy here — in a readmission-screening context, a missed at-risk patient is more costly than a false positive that just triggers extra follow-up. LogReg catches more than twice as many true readmissions (69% vs. 31%).

**"Why is the model's F1 score so low?"**
Small training data (275 admissions) and a genuinely hard, imbalanced prediction problem (only ~19% of admissions are readmissions). The honest framing is that this demonstrates correct, leakage-free methodology — the ceiling on performance is a data-size problem, not a methodology problem.

**"How did you avoid data leakage?"**
By using *prior* admissions count (known at admission time) instead of each patient's lifetime total admissions (which isn't known until after their last visit), and by using only admission-time features — age, diagnosis count, prior admissions, admission type.

**"Is the medication data connected to these patients?"**
No — deliberately not. OpenFDA is a public, independent adverse-event feed. It answers "which medications look risky in general," not "which of these 100 patients had an adverse event." That's stated as a limitation, not glossed over.

**"Could this go into production?"**
Not as-is — it's a proof of concept on a small demo dataset. The path there is exactly what's on the recommendations slide: a larger prospective cohort, linked medication history, and ongoing performance monitoring.

**"What would you do differently with more time?"**
Link medication and clinical data directly, test additional features (e.g., lab values, vitals), and validate the risk thresholds against clinical judgment rather than a fixed statistical rule.
