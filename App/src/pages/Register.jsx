import { Link } from 'react-router-dom';

function Register() {
  return (
    <section className="panel auth-panel">
      <header>
        <p className="eyebrow">Get started</p>
        <h2>Create your intelligence workspace</h2>
        <p>Set up credentials so you can upload datasets, run comparisons, and share insights with reviewers.</p>
      </header>

      <form className="auth-form">
        <label>
          Full name
          <input type="text" placeholder="Aarav Sharma" required />
        </label>
        <label>
          Institutional email
          <input type="email" placeholder="name@college.edu" required />
        </label>
        <label>
          Create password
          <input type="password" placeholder="Use 8+ characters" required />
        </label>
        <label>
          Team code (optional)
          <input type="text" placeholder="e.g. INTEL2025" />
        </label>
        <div className="form-actions">
          <button type="submit" className="primary">
            Register
          </button>
          <Link to="/login">Already registered? Log in</Link>
        </div>
      </form>
    </section>
  );
}

export default Register;

