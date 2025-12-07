const solutions = [
  {
    title: 'Salary Intelligence Studio',
    description:
      'Upload cleaned datasets, choose regression recipes, and compare MAE/RMSE before exporting actionable pay bands.',
    checklist: ['Role-based benchmarking', 'Explainable feature weights', 'Excel & JSON exports'],
    image:
      'https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&w=900&q=80',
  },
  {
    title: 'Fake Post Radar',
    description:
      'Our competitive classifiers vet posts in seconds, leveraging NLP embeddings, link validation, and profile scoring.',
    checklist: ['Confidence badges for each listing', 'Escalation workflow', 'Teacher review dashboard'],
    image:
      'https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?auto=format&fit=crop&w=900&q=80',
  },
  {
    title: 'Insight Briefings',
    description:
      'Beautiful story-driven slides summarise market signals, ready for academic juries or partner organizations.',
    checklist: ['Auto-generated highlights', 'Shareable reports', 'Live data refresh hooks'],
    image:
      'https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=900&q=80',
  },
];

function Solutions() {
  return (
    <section className="panel content-panel">
      <header>
        <p className="eyebrow">What we deliver</p>
        <h2>Competitive intelligence you can demo today</h2>
        <p>
          Each module demonstrates applied research: from data ingestion to insights. Teachers can click through these
          cards to understand scope before diving into the backend notebooks.
        </p>
      </header>

      <div className="solutions-grid">
        {solutions.map((solution) => (
          <article key={solution.title}>
            <img src={solution.image} alt={solution.title} />
            <h3>{solution.title}</h3>
            <p>{solution.description}</p>
            <ul>
              {solution.checklist.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </article>
        ))}
      </div>
    </section>
  );
}

export default Solutions;

