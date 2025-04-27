import { useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Menu, X, User } from "lucide-react";
import AuthModal from "@/components/auth/AuthModal";
import { useNavigate } from "react-router-dom";

export default function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const navigate = useNavigate();

  const handleLogin = () => {
    setAuthMode("login");
    setAuthModalOpen(true);
  };

  const handleRegister = () => {
    navigate("/register");
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
  };

  const onAuthSuccess = () => {
    setIsAuthenticated(true);
    setAuthModalOpen(false);
  };

  return (
    <nav className="bg-soccer-blue py-4 sticky top-0 z-30">
      <div className="container mx-auto flex justify-between items-center">
        <div className="flex items-center">
          <Link to="/" className="text-white font-bold text-2xl tracking-tight hover:text-green-200">
            TheDrawCode
          </Link>
        </div>

        <div className="hidden md:flex space-x-6 items-center">
          <Link to="/matches" className="text-white hover:text-green-200 font-medium">
            Matches
          </Link>
          <Link to="/leagues" className="text-white hover:text-green-200 font-medium">
            Leagues
          </Link>
          <Link to="/about" className="text-white hover:text-green-200 font-medium">
            About
          </Link>
          
          {isAuthenticated ? (
            <div className="flex items-center space-x-4">
              <Link to="/dashboard" className="text-white hover:text-green-200 font-medium">
                Dashboard
              </Link>
              <Button 
                variant="outline" 
                className="text-white hover:bg-soccer-green border-soccer-green"
                onClick={handleLogout}
              >
                <User className="mr-2 h-4 w-4" /> Account
              </Button>
            </div>
          ) : (
            <div className="flex items-center space-x-3">
              <Button 
                variant="ghost" 
                className="text-white hover:text-white hover:bg-soccer-blue-light"
                onClick={handleLogin}
              >
                Login
              </Button>
              <Button 
                className="bg-soccer-green hover:bg-soccer-green/90 text-white"
                onClick={handleRegister}
              >
                Sign Up
              </Button>
            </div>
          )}
        </div>

        <button
          className="md:hidden text-white"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        >
          {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {mobileMenuOpen && (
        <div className="container mx-auto md:hidden mt-4 pb-4 flex flex-col space-y-4">
          <Link 
            to="/matches" 
            className="text-white hover:text-green-200 font-medium py-2"
            onClick={() => setMobileMenuOpen(false)}
          >
            Matches
          </Link>
          <Link 
            to="/leagues" 
            className="text-white hover:text-green-200 font-medium py-2"
            onClick={() => setMobileMenuOpen(false)}
          >
            Leagues
          </Link>
          <Link 
            to="/about" 
            className="text-white hover:text-green-200 font-medium py-2"
            onClick={() => setMobileMenuOpen(false)}
          >
            About
          </Link>
          
          {isAuthenticated ? (
            <>
              <Link 
                to="/dashboard" 
                className="text-white hover:text-green-200 font-medium py-2"
                onClick={() => setMobileMenuOpen(false)}
              >
                Dashboard
              </Link>
              <Button 
                variant="outline" 
                className="text-white hover:bg-soccer-green border-soccer-green justify-start"
                onClick={() => {
                  handleLogout();
                  setMobileMenuOpen(false);
                }}
              >
                <User className="mr-2 h-4 w-4" /> Account
              </Button>
            </>
          ) : (
            <div className="flex flex-col space-y-2">
              <Button 
                variant="ghost" 
                className="text-white hover:text-white hover:bg-soccer-blue-light justify-start"
                onClick={() => {
                  handleLogin();
                  setMobileMenuOpen(false);
                }}
              >
                Login
              </Button>
              <Button 
                className="bg-soccer-green hover:bg-soccer-green/90 text-white justify-start"
                onClick={() => {
                  handleRegister();
                  setMobileMenuOpen(false);
                }}
              >
                Sign Up
              </Button>
            </div>
          )}
        </div>
      )}

      <AuthModal 
        isOpen={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        initialMode={authMode}
        onSuccess={onAuthSuccess}
      />
    </nav>
  );
}
