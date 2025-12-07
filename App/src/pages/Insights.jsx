const insights = [
  {
    metric: '68%',
    title: 'Listings flagged as risky',
    detail: 'Model ensemble cross-validates each job post to catch scam patterns and missing employer signals.',
  },
  {
    metric: '₹18.6L',
    title: 'Median salary prediction',
    detail: 'Regression stack blends LightGBM + Random Forest to output interpretable salary ranges.',
  },
  {
    metric: '4 hrs',
    title: 'Time saved per review',
    detail: 'Automated vetting lets mentors focus on strategy instead of manual due diligence.',
  },
];

const insightMedia = [
  {
    title: 'Model comparison slide',
    copy: 'Snapshot of our competition notebook summarizing MAE, RMSE, and top hyperparameters.',
    image:
      'https://images.unsplash.com/photo-1454165205744-3b78555e5572?auto=format&fit=crop&w=900&q=80',
  },
  {
    title: 'Field research capture',
    copy: 'Gallery view of the fake-post cues collected from public reports and recruiter interviews.',
    image:
      'https://images.unsplash.com/photo-1529333166437-7750a6dd5a70?auto=format&fit=crop&w=900&q=80',
  },
];

function Insights() {
  return (
    <section className="panel content-panel">
      <header>
        <p className="eyebrow">Data Stories</p>
        <h2>Market pulse & platform promises</h2>
        <p>
          We track the health of the job market using competition-ready models trained on curated datasets. These stats
          help stakeholders understand why our dual focus on salary and authenticity matters.
        </p>
      </header>

      <div className="metrics-grid">
        {insights.map((item) => (
          <article key={item.title}>
            <span className="metric">{item.metric}</span>
            <h3>{item.title}</h3>
            <p>{item.detail}</p>
          </article>
        ))}
      </div>

      <div className="insight-media">
        {insightMedia.map((media) => (
          <article key={media.title}>
            <img src={media.image} alt={media.title} />
            <div>
              <h4>{media.title}</h4>
              <p>{media.copy}</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

export default Insights;

