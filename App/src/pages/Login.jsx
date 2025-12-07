import { Link } from 'react-router-dom';

function Login() {
  return (
    <section className="panel auth-panel">
      <header>
        <p className="eyebrow">Welcome back</p>
        <h2>Log in to continue analysis</h2>
        <p>Access the salary intelligence studio, monitor fake posts, and pick up right where you left off.</p>
      </header>

      <form className="auth-form">
        <label>
          Email address
          <input type="email" placeholder="you@example.com" required />
        </label>
        <label>
          Password
          <input type="password" placeholder="••••••••" required />
        </label>
        <div className="form-actions">
          <button type="submit" className="primary">
            Log in
          </button>
          <Link to="/register">Need an account? Register</Link>
        </div>
      </form>
    </section>
  );
}

export default Login;

