import { Link, NavLink, Outlet } from 'react-router-dom';

const navItems = [
  { to: '/', label: 'Overview' },
  { to: '/insights', label: 'Market Insights' },
  { to: '/solutions', label: 'Solutions' },
  { to: '/login', label: 'Login' },
  { to: '/register', label: 'Register', highlight: true },
];

function Layout() {
  return (
    <div className="layout">
      <header className="nav-bar">
        <Link to="/" className="logo">
          Job Market Intelligence
        </Link>
        <nav>
          <ul>
            {navItems.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  className={({ isActive }) =>
                    ['nav-link', isActive ? 'active' : '', item.highlight ? 'nav-cta' : ''].join(' ').trim()
                  }
                  end={item.to === '/'}
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </header>

      <div className="page-container">
        <Outlet />
      </div>

      <footer className="site-footer">
        <p>© {new Date().getFullYear()} Job Market Intelligence · Salary & Fake Post Prediction Lab</p>
      </footer>
    </div>
  );
}

export default Layout;

