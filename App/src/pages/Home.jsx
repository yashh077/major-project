import { Link } from 'react-router-dom';
import heroImage from '../assets/hero-placeholder.svg';

const highlights = [
  {
    title: 'Salary Intelligence',
    body: 'Benchmark compensation across roles, skills, and locations using ensemble regression pipelines.',
  },
  {
    title: 'Fake Post Prediction',
    body: 'Flag suspicious job listings with our competitive classification models before they reach candidates.',
  },
  {
    title: 'Human-Friendly Reporting',
    body: 'Explainable charts and narrative summaries make it easy to brief faculty or industry mentors.',
  },
];

const referenceShots = [
  {
    title: 'Compensation Radar',
    description: 'Visual mock showing salary bands stacked against demanded skill sets.',
    image:
      'https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&w=900&q=80',
  },
  {
    title: 'Fraud Pattern Board',
    description: 'Research wall highlighting scam cues gathered from public job boards.',
    image:
      'https://images.unsplash.com/photo-1545239351-1141bd82e8a6?auto=format&fit=crop&w=900&q=80',
  },
  {
    title: 'Analyst Handoff Kit',
    description: 'Workspace dashboard that supervisors will see during the demo.',
    image:
      'https://images.unsplash.com/photo-1522199710521-72d69614c702?auto=format&fit=crop&w=900&q=80',
  },
];

function Home() {
  return (
    <section className="panel hero-panel">
      <div className="hero-copy">
        <p className="eyebrow">Major Project 2025</p>
        <h1>Job Market Intelligence Portal</h1>
        <p>
          We combine salary forecasting and fake post detection to create a trustworthy experience for job seekers and
          institutions. Today&apos;s web preview walks through the experience teachers will evaluate: clean navigation,
          clear storytelling, and the entry points for analytical workflows.
        </p>
        <div className="hero-actions">
          <Link className="primary" to="/solutions">
            Explore Solutions
          </Link>
          <Link className="ghost" to="/register">
            Create Account
          </Link>
        </div>
      </div>
      <img src={heroImage} alt="Illustration for market intelligence" />

      <div className="highlight-grid">
        {highlights.map((item) => (
          <article key={item.title}>
            <h3>{item.title}</h3>
            <p>{item.body}</p>
          </article>
        ))}
      </div>

      <div className="reference-gallery">
        {referenceShots.map((shot) => (
          <figure key={shot.title}>
            <img src={shot.image} alt={shot.title} />
            <figcaption>
              <h4>{shot.title}</h4>
              <p>{shot.description}</p>
            </figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}

export default Home;

