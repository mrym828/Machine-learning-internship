# دليل مرجعي شامل — كل المشروع (w01 إلى w07) / Full Project Study Guide

> هذا ملف مرجعي شخصي لي بس، للمراجعة، مو جزء من التسليم الرسمي. الدفاتر الرسمية بـ`work/notebooks/`.
> This is a personal reference file for my own review only — not a graded deliverable. The official notebooks live in `work/notebooks/`.
>
> يغطي هذا الملف كل أسبوع من w01 إلى w07 بالعربي والإنجليزي، مع الكود الحقيقي والأرقام الحقيقية اللي طلعت.
> This file covers every week from w01 through w07, in both Arabic and English, with the real code and real numbers produced.

---

## الفكرة العامة للمشروع كاملاً / The Whole Project, Big Picture

**بالعربي:** المشروع كامل يجاوب على سؤال واحد: "شنو أنماط الأداء (archetypes) الموجودة بين صفحات المحتوى، وأي صفحة يستاهل أخصائي SEO يراجعها أول؟" اخترتي مسار **التجميع (clustering)** — يعني ما فيه إجابة جاهزة، النظام يكتشف المجموعات بنفسه من سلوك الصفحات الفعلي (نسبة الضغط، الترتيب في جوجل).

**English:** The whole project answers one question: "What performance archetypes exist among content pages, and which page should an SEO specialist review first?" The lane chosen is **clustering** — unsupervised, no ready-made answer; the model discovers groups from real page behavior (click-through rate, Google position).

**المسار الأسبوعي / Weekly path:**

```
w01 سؤال البحث → w02 تأطير كـ ML → w03 عقد البيانات → w04 القاعدة الأساسية
→ w05 النموذج الحقيقي (K-Means) → w06 تدقيق الصحة → w07 خطة العمل → capstone الورقة النهائية
```

---

## w01 — Research Question (ML-02)

### بالعربي
- **اخترتي الـ lane:** Structured Content Archetype Clustering — لأن `engagement_rate` عند أغلب الصفحات قريب من الصفر، بس فيه مجموعة صغيرة أداءها أعلى بكثير، دليل إن المحتوى خليط من أنواع مختلفة.
- **القرار اللي المشروع يخدمه:** أخصائي SEO يقرر أي صفحة يراجعها أول. الخطأ له تكلفتين: صفحة تحتاج شغل واتحسبت تمام = خسارة زيارات ممكنة، صفحة تمام واتحسبت تحتاج شغل = وقت ضايع.
- **الكلام المسموح:** "لاحظنا" مجموعات بأداء مختلف — decision-support مو إثبات سببي.

### English
- **Lane chosen:** engagement metrics are heavily skewed — most pages near zero, a small group performing much better.
- **Decision served:** which pages an SEO specialist reviews first. Cost of a wrong call cuts both ways.
- **Careful words:** "observed," "directional," "decision-support" — never a causal claim.

### Code
```python
import pandas as pd
url = "https://raw.githubusercontent.com/mrym828/Machine-learning-internship/main/data/raw/content_refresh_anonymized.csv"
df = pd.read_csv(url)
df['content_type'].value_counts()
df['engagement_rate'].describe()
```
`engagement_rate`: mean 2.53, median 0.0 — most pages near-dead, a minority pulls the average up.

---

## w02 — Frame as an ML Task (ML-03)

### بالعربي
- **نوع المهمة:** Clustering — نكتشف مجموعات، مو نتنبأ بإجابة معروفة.
- **البديل عن الهدف:** ما فيه label حقيقي، البديل هو نفس الخصائص الخمس (engagement_rate, scroll_rate, ctr, avg_position, ai_traffic_pct).
- **مقياس النجاح:** silhouette score + فحص بشري.
- **ليش ML أفضل من قاعدة بسيطة:** قاعدة زي "لو engagement أقل من 1 يبقى صفحة ميتة" ما تلتقط التفاعل بين كل الخصائص مع بعض.

### English
- **ML task type:** Clustering (unsupervised).
- **Target/proxy:** no real label; the feature set itself is the proxy.
- **Success metric:** silhouette score, balanced against human sense-checking.
- **Unit of analysis:** one row = one content item.

### Code
```python
cols = ["engagement_rate", "scroll_rate", "ctr", "avg_position", "ai_traffic_pct"]
cluster = df[["content_id"] + cols]
```
Noted data quirk: many rows have `avg_position = 0`, meaning "no position data," not a real rank of zero.

---

## w03 — Data Contract (ML-04)

### بالعربي
هذا الأسبوع انتقلنا من الـ CSV الصغير إلى الـ **warehouse** (قاعدة بيانات ضخمة على Hugging Face). كتبنا "عقد بيانات": وثيقة تشرح بالضبط شنو معنى كل صف، وأي أعمدة نستخدم، ونثبت كل شي بكود فعلي.

### English
Moved from the small starter CSV to the **warehouse** (huge dataset on Hugging Face). Wrote a data contract — a verified promise about what the data means before touching any model.

### الاتصال بالـ warehouse / Connecting via DuckDB
```python
from dotenv import load_dotenv
import os, duckdb

load_dotenv()
token = os.environ["HF_TOKEN"]
con = duckdb.connect()
con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{token}')")

FACT = "read_parquet('hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance/**/*.parquet')"
CONTENT = "read_parquet('hf://datasets/FlyRank/internship-warehouse/dim_content.parquet')"
```
**مهم:** التوكن دايم من `.env` (محلي، مو مرفوع على git)، أبداً ما يُكتب مباشرة بالكود.
**Important:** the token always comes from `.env` (local, gitignored), never pasted directly in a cell.

### وحدة التحليل والنافذة الزمنية / Unit of analysis + window
**بالعربي:** صف واحد = صفحة محتوى وحدة، نشاطها **مجموع على مارس 2026 بس**، اخترنا مارس لأنه شهر "وسط" (مو أول ولا آخر شهر).
**English:** one row = one content item, activity summed over **March 2026 only**, chosen as a mid-panel month for stable coverage.

### تصنيف الأعمدة / Field classification
- **Feature:** `ctr`, `avg_position` (primary), plus GA4-based secondary features (excluded later due to thin coverage).
- **Context:** all `*_hash_id` columns, `report_date`/`month`.
- **Excluded:** `provider_used`/`model_used` (not behavior signals); per-channel session splits (didn't reconcile with totals when checked — a real finding, not assumed).

### الاكتشاف الكبير: تغطية البيانات ضعيفة / The big finding: coverage is thin
```python
con.sql(f"""
    SELECT COUNT(DISTINCT content_hash_id) AS total_items,
           COUNT(DISTINCT content_hash_id) FILTER (WHERE gsc_data_available IS TRUE) AS items_with_gsc,
           COUNT(DISTINCT content_hash_id) FILTER (WHERE ga4_data_available IS TRUE) AS items_with_ga4
    FROM {FACT} WHERE month = '2026-03'
""").show()
```
**النتيجة:** بس **53.3%** من الصفحات عندها بيانات GSC حقيقية بمارس، وبس **27.3%** عندها بيانات GA4 حقيقية.
**Result:** only **53.3%** of items have real GSC coverage in March, only **27.3%** have real GA4 coverage.

**القرار:** الخصائص الأساسية (ctr, avg_position) تُقاس بس على الصفحات اللي عندها تتبع GSC حقيقي. خصائص GA4 صارت تحليل ثانوي بس.
**Decision:** primary features (ctr, avg_position) restricted to real GSC coverage; GA4 features became a secondary, smaller-scope analysis.

### فخ التسريب (leakage trap) / The leakage trap demo
جربنا فخين "بريئين" ما اشتغلوا زي المتوقع (نتيجة صادقة، مو ملفقة):
Two "innocent-looking" leak attempts that did NOT work as expected (honest result, not forced):
- binned version of `avg_position` → silhouette **dropped** from 0.724 to 0.694
- exact duplicate of `avg_position` → dropped further to 0.667

الفخ الحقيقي: تغذية تسميات الكتلة نفسها كخاصية (دائرية حقيقية):
The real trap: feeding the cluster labels back in as a feature (genuine circularity):
```python
features['cluster_label_leak'] = labels  # from the honest baseline fit
```
النتيجة قفزت لـ **0.828** (تضخم كاذب). الرقم الصادق المحتفظ فيه: **0.724**.
Score jumped to **0.828** (fake inflation). The honest number kept: **0.724**.

---

## w04 — Baseline Action Score (ML-07)

### بالعربي
بنينا "القاعدة البسيطة" اللي أي نموذج لازم يتفوق عليها. القاعدة قابلة للقراءة من إنسان عادي، مو صندوق أسود.

### English
Built the transparent, rule-based baseline every model must beat. Readable by a human, not a black box.

### فحص الإشارات قبل بناء القاعدة / Signal checks before building the rule
**إشارة 1 — القدم (staleness):** الفكرة: صفحة ما تحدثت من زمن أداءها أسوأ.
```python
visible = df[df['impressions_90d'] >= 500].copy()
visible['is_stale'] = visible['days_since_last_update'] >= 180
bucket_staleness = visible.groupby('is_stale').agg(n=('content_id','count'), mean_ctr=('ctr','mean'), mean_engagement_rate=('engagement_rate','mean')).reset_index()
```
**النتيجة: MIXED** — بس 17 صف من 30 ألف قدموا، والاتجاهين متناقضين (ctr أسوأ بس engagement أحسن). ما بنينا القاعدة على هذي الإشارة.
**Result: MIXED** — only n=17 out of 30,000, and the two metrics disagreed. Did not build the rule on this signal.

**إشارة 2 — CTR مقابل الترتيب:** النتيجة **CONFIRMED** — CTR ينزل بانتظام كل ما الترتيب يسوء، بس الحد الثابت (0.5%) طلع يطبق على 75% حتى بأفضل المراتب — يعني لازم مقارنة نسبية لكل فئة ترتيب، مو رقم ثابت.
**Signal 2 — CTR vs position: CONFIRMED** — CTR drops monotonically with worse position, but the flag's fixed 0.5% threshold caught 75% of pages even at the best positions — needed a tier-relative comparison instead.

### القاعدة والقائمة المرتبة / The rule and the ranked queue
```python
tier_median_ctr = visible.groupby('position_tier')['ctr'].median().rename('tier_median_ctr')
df = df.merge(tier_median_ctr, on='position_tier', how='left')
df['underperforms_ctr'] = ((df['position_tier'] != 'no_data') & (df['ctr'] < 0.7 * df['tier_median_ctr'])).astype(int)
df['baseline_action_score'] = df['underperforms_ctr'] * df['is_visible'] * df['impressions_90d']
df['baseline_rank'] = df['baseline_action_score'].rank(method='first', ascending=False).astype(int)
```
**النتيجة:** 6,058 صفحة من 30,000 انطبق عليها الشرط (~20%).
**Result:** 6,058 of 30,000 pages flagged (~20%).

### نقاط ضعف حقيقية اكتشفناها / Real weak points found
مراجعة أول 10 صفوف بأيدينا كشفت: صفوف بفجوة CTR صغيرة (33% تحت المتوسط، بالكاد تعدت حد التأهل 30%) طلعت بترتيب أعلى من صفوف بفجوة أكبر بكثير (85%)، بس لأن الدرجة = flag × impressions، فالدرجة تكافئ الحجم مو شدة المشكلة.
Hand-reviewing the top 10 revealed: rows with a small CTR gap (33% below median, barely past the 30% qualifying line) ranked higher than rows with a much bigger gap (85%), because the score rewards size, not severity.

---

## w05 — The Real Model: K-Means Clustering (ML-08)

### بالعربي
هذا الأسبوع بنينا النموذج الحقيقي (K-Means) على خصائص w03's contract، وقارنّاه بقاعدة عشوائية، وتحققنا من ثباته على عملاء ما شافهم النموذج.

### English
Built the real model (K-Means) on w03's contract features, compared against a random dummy baseline, and validated stability on unseen clients.

### فهم K-Means من الصفر / Understanding K-Means from scratch
تعلمنا الآلية بمثال بسيط (6 زبائن، إنفاق وزيارات) قبل البيانات الحقيقية:
Learned the mechanism with a toy example (6 customers, spend & visits) before touching real data:
```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, random_state=42, n_init=10)
customers['cluster'] = km.fit_predict(customers[['spend', 'visits']])
```
- **Centroid** = متوسط كل نقطة بالمجموعة (حسبناها يدوياً وطابقت الكود).
- **Centroid** = average of every point in the group (hand-calculated, matched the code).
- **ليش نطبّع (scale) قبل التجميع:** بدون تطبيع، الرقم الأكبر (زي spend بالآلاف) يطغى على الرقم الأصغر (rating من 1-5) حتى لو كان الأصغر أهم.
- **Why scale first:** without it, the bigger-magnitude column (spend) dominates distance math even if the smaller column (rating) is more meaningful.

### بناء القاعدة الحقيقية / Building the real pipeline
```python
features = con.sql(f"""
    SELECT content_hash_id, SUM(gsc_clicks) AS clicks_sum, SUM(gsc_impressions) AS impressions_sum, AVG(gsc_avg_position) AS avg_position
    FROM {FACT} WHERE month = '2026-03' AND gsc_data_available IS TRUE GROUP BY content_hash_id
""").df()
features['ctr'] = features['clicks_sum'] / features['impressions_sum']
```

### الفخ الأول: نتيجة "مثالية جداً" / Trap 1: a "too perfect" score
k=2 أعطى silhouette = **0.94** (يبدو رائع!). لكن بدل الاحتفال، فحصنا: طلع كتلة وحدة 279 صف بس، كلها صفحات بـ1-2 impressions حيث ضغطة واحدة بالحظ تعطي CTR قريب من 100%. **مو أرشيتايب حقيقي، ضجيج من عينة صغيرة.**
k=2 gave silhouette = **0.94** (looks amazing!). Instead of celebrating, we investigated: one cluster of just 279 rows, all pages with 1-2 impressions where a single lucky click created a near-100% CTR. **Not a real archetype — small-sample noise.**

**الحل:** حد أدنى `impressions_sum >= 30` قبل التجميع.
**Fix:** a volume floor of `impressions_sum >= 30` before clustering.

### اختيار k الصحيح / Choosing the real k
```python
for k in [2, 3, 4, 5, 6]:
    km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km_test.fit_predict(X_train)
    score = silhouette_score(X_train, labels, sample_size=5000, random_state=42)
```
بعد التنظيف: k=3 كان الأوضح (~0.52-0.62 حسب الدفعة)، مو k=2.
After cleaning: k=3 was the clear winner (~0.52-0.62 depending on the run), not k=2.

### المقارنة مع القاعدة العشوائية والاستقرار / Comparison vs dummy baseline + stability
```python
random_labels = rng2.integers(0, 3, size=X_train.shape[0])
dummy_score = silhouette_score(X_train, random_labels, sample_size=5000, random_state=42)  # ≈ -0.01 إلى -0.02, صفر تقريباً
```
| الطريقة / method | silhouette |
|---|---|
| dummy (عشوائي) | ≈ -0.01 to -0.02 |
| K-Means (train) | ~0.52-0.62 |
| K-Means (test, عملاء ما شافهم النموذج) | ~0.53-0.64 |

**الأرشيتايبات الثلاثة المسماة / The three named archetypes:**
- **Champions** — ترتيب زين + CTR زين
- **Underperforming Good Position** — ترتيب زين بس CTR أسوأ 10 مرات
- **Buried and Ignored** — ترتيب سيء + CTR سيء

### بق حقيقي في إعادة الإنتاج / A real reproducibility bug
```python
unique_clients = features['client_hash_id'].unique()  # ترتيب الصفوف من SQL مو مضمون!
rng = np.random.default_rng(42)
shuffled_clients = rng.permutation(unique_clients)
```
كل تشغيلة أعطت تقسيم مختلف رغم الـ seed الثابت! السبب: استعلام SQL ما يضمن نفس ترتيب الصفوف، فـ`.unique()` يرجع ترتيب مختلف، فالخلط (`permutation`) يعطي نتيجة مختلفة.
Every run gave a different split despite a fixed seed! Because the SQL query doesn't guarantee row order, so `.unique()` returns clients in a different order each time, changing the shuffle result.
**الحل / Fix:** `sorted(features['client_hash_id'].unique())` — ترتيب أبجدي ثابت قبل الخلط. **نفس البق طلع لاحقاً في w07 أيضاً وانصلح بنفس الطريقة (ترتيب الصفوف بـ content_hash_id قبل أي fit).**
**The same bug reappeared later in w07 too, fixed the same way (sort rows by content_hash_id before any fit).**

---

## w06 — Validation and Research Claim Audit (ML-09)

### بالعربي
هذا الأسبوع سويتي شيئين: (أ) راجعتي ورقة بحث FlyRank الحقيقية بنفس معايير الصرامة، و(ب) طبقتي نفس المعايير على شغلك أنتِ.

### English
Two things this week: (a) audited FlyRank's real research paper with the same rigor, and (b) turned that same lens on your own work.

### فحص ورقة البحث / Auditing the research paper
اخترتي اكتشافين وسألتي سؤال منهجية بناء لكل وحد:
Picked two findings and asked a constructive methodology question for each:
1. **Feature Importance for Health Score** — `health_score` معرّف جزئياً من `position`/`impressions`/`ctr`، فاستخدام Random Forest ليتنبأ بيه من نفس الأعمدة شبه دائري (label-derived feature trap).
   `health_score` is partly built from position/impressions/ctr, so predicting it from the same columns is close to circular.
2. **Freshness Multiplier / refresh effect** — "3.2x health boost, 57x impressions" من مقارنة رصدية بدون مجموعة ضبط. سؤال الانحياز: هل المحررين اختاروا صفحات واعدة أصلاً؟
   "3.2x health boost, 57x impressions" from an observational comparison, no control group — selection-bias question about whether editors chose already-promising pages.

### نموذجك أنتِ تحت تقسيم صادق (قبل/بعد) / Your model under an honest split (before/after)
```python
naive_train, naive_test = train_test_split(features, test_size=0.5, random_state=42)  # قبل: عشوائي بدون مراعاة العميل
```
| split | train | test | الفجوة/gap |
|---|---|---|---|
| naive random (before) | 0.5996 | 0.5980 | 0.0016 |
| grouped by client (after, honest) | 0.6156 | 0.5354 | 0.0802 |

**الاكتشاف الحقيقي:** التقسيم العشوائي يخلط كل العملاء مع بعض ويخفي الفروقات الحقيقية بينهم؛ التقسيم حسب العميل يكشفها. الفجوة الأكبر مو مشكلة، هي دليل صحة.
**The real finding:** random splitting blends every client together and hides real differences; client-grouped splitting reveals them. The bigger gap is evidence of honesty, not a problem.

### تدقيق التسريب / Leakage audit
أعدنا فخ w03 (تغذية تسمية الكتلة كخاصية) على النموذج النهائي: 0.6156 → **0.6615** — القفزة أصغر من w03 الأصلية بس بنفس الاتجاه، يثبت إن أداة الفحص شغالة صح.
Re-ran w03's trap (feed cluster label back as a feature) on the final model: 0.6156 → **0.6615** — smaller jump than w03's original, but same direction, confirming the test harness works.

### إعادة صياغة الادعاء الأجرأ / Claim rewrite
الجملة الأصلية استخدمت "prove" و"need" (لغة سببية) على بيانات رصدية بس. أعيدت صياغتها لتقول "directional evidence, not proof" و"one plausible next step to test."
The original sentence used "prove" and "need" (causal language) on purely observational data. Rewritten to say "directional evidence, not proof."

---

## w07 — Content Action Playbook (ML-10)

### بالعربي
حولنا الأرشيتايبات المتحقق منها لقائمة عمل فعلية: تصنيف، سبب، إجراء لكل صفحة.

### English
Turned the validated archetypes into an actual action queue: archetype, reason code, action per page.

### الخريطة الديناميكية / Dynamic archetype mapping
```python
champions_cluster = profile.loc[profile['mean_ctr'].idxmax(), 'cluster']
buried_cluster = profile.loc[profile['mean_position'].idxmax(), 'cluster']
```
مو hardcoded index، لأن ترقيم K-Means للكتل مو ثابت بين التشغيلات.
Not a hardcoded index, since K-Means cluster numbering isn't guaranteed stable across runs.

**الأرقام النهائية المتحقق من ثباتها (بعد إصلاح بق الترتيب) / Final, reproducibility-verified numbers (after fixing the row-order bug):**

| الأرشيتايب | reason_code | action | n | % |
|---|---|---|---|---|
| Champions | champion_maintain | protect_and_monitor | 5,831 | 4.64% |
| Underperforming Good Position | ctr_below_expected | refresh_and_review_ctr | 97,799 | 77.84% |
| Buried and Ignored | buried_low_priority | deprioritize_or_prune_review | 22,015 | 17.52% |

### بق حقيقي في الترتيب / A real ranking bug
```python
features['playbook_rank'] = features.sort_values([...]).reset_index().index + 1  # خطأ!
```
`.reset_index()` يبني DataFrame جديد منفصل؛ لما نحاول ننسب `.index` رجوع لـ `features`، pandas يطابق حسب رقم الصف الأصلي مو الترتيب الصحيح — فتختلط الأرقام.
`.reset_index()` builds a separate new frame; assigning its `.index` back into `features` makes pandas align by original row label, not the correct sorted position — scrambling the ranks.
**الحل / Fix:**
```python
features = features.sort_values([...]).reset_index(drop=True)
features['playbook_rank'] = features.index + 1
```

### القيود ولا-تفعل / Limits and no-go list
- ما يشمل إلا صفحات عندها تغطية GSC حقيقية بمارس (~53%).
- لقطة شهر وحد بس، الأرشيتايب ممكن يتغير الشهر الجاي.
- ممنوع: حذف تلقائي، تعديل محتوى تلقائي، الحكم على الكاتب من الأرشيتايب.
- No auto-delete, no auto content edits, no judging the writer from the archetype.

### التصدير / Exports
```python
ranked_queue.to_csv(output_dir / "action_playbook_queue.csv", index=False)  # خارج git
json.dump(metadata, open(output_dir / "playbook_metadata.json", "w"))       # داخل git
fig.savefig(figures_dir / "archetype_scatter.png", dpi=150)                  # داخل git
```

---

## capstone.ipynb — تجميع كل شي / Assembling everything

**بالعربي:** الدفتر الأخير يعيد بناء نفس الأنبوب (data → methodology → results → recommendations) بشكل مستقل وقابل لإعادة التشغيل، بالأرقام النهائية المتحقق منها: dummy baseline ≈ -0.015، K-Means train = 0.624، test (عملاء ما شافهم النموذج) = 0.529.

**English:** The final notebook rebuilds the same pipeline (data → methodology → results → recommendations) independently and reproducibly, with the final verified numbers: dummy baseline ≈ -0.015, K-Means train = 0.624, test (unseen clients) = 0.529.

**بق اكتشفناه أثناء البناء / A bug caught while building it:** كتبت نص markdown داخل خلايا كانت لسا نوعها "code" بدون ما أحدد `cell_type=markdown`، فحاول Jupyter ينفذ النص كبايثون وطلع SyntaxError. الحل: تحديد `cell_type` بوضوح عند تعديل خلايا موجودة مسبقاً.
Markdown prose got written into cells that were still `code`-typed, without explicitly setting `cell_type=markdown`, so Jupyter tried to execute the prose as Python and threw a SyntaxError. Fix: always specify `cell_type` explicitly when editing pre-existing cells.

---

## مسرد المصطلحات / Glossary

| English | بالعربي | معناها |
|---|---|---|
| Clustering | تجميع | اكتشاف مجموعات بدون إجابة جاهزة |
| Centroid | مركز الكتلة | متوسط كل النقاط بالمجموعة |
| Silhouette score | مقياس السيلويت | مدى تماسك كل مجموعة وابتعادها عن الباقي |
| Dummy baseline | قاعدة عشوائية | أضعف مقارنة ممكنة، "الأرضية تحت الأرضية" |
| Data leakage | تسريب البيانات | معلومة من الإجابة تدخل بالغلط كخاصية |
| Grouped split | تقسيم مجمّع | تقسيم بيانات حسب كيان متكرر (عميل) مو صفوف عشوائية |
| Reproducibility | إمكانية إعادة الإنتاج | نفس الكود ونفس البذرة يعطي نفس النتيجة دايماً |
| Reason code | رمز السبب | كلمة قصيرة تفسر ليش صفحة انحطت بهذا الترتيب |
| Volume floor | حد أدنى للحجم | أقل عدد impressions نثق فيه قبل حساب نسبة |
| Claim ladder | سلم الادعاءات | observed → directional → decision-support → causal (بس بتجربة حقيقية) |

---

## الورقة البحثية النهائية / The Final Research Paper

نُشرت الورقة كصفحة ويب عامة (`docs/index.html`) تجمع كل هذا: العنوان، الملخص، السؤال، البيانات، المنهجية، النتائج، القيود، التوصيات المرتبة، إمكانية إعادة الإنتاج، والشكر + رابط مصدر البيانات (flyrank.ai).
The paper was published as a public web page (`docs/index.html`) assembling all of this: title, abstract, question, data, methodology, results, limitations, ranked recommendations, reproducibility, and acknowledgments + data credit (flyrank.ai).
