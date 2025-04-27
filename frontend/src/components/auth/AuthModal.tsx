
import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import LoginForm from "./LoginForm";
import RegisterForm from "./RegisterForm";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialMode: "login" | "register";
  onSuccess: () => void;
}

export default function AuthModal({ isOpen, onClose, initialMode, onSuccess }: AuthModalProps) {
  const [mode, setMode] = useState<"login" | "register" | "forgot-password">(initialMode);
  const [email, setEmail] = useState("");
  const [resetSent, setResetSent] = useState(false);

  const handleModeSwitch = (newMode: "login" | "register" | "forgot-password") => {
    setMode(newMode);
  };

  const handleForgotPassword = async () => {
    try {
      await API.auth.resetPassword(email);
      setResetSent(true);
    } catch (error) {
      console.error("Error sending password reset:", error);
    }
  };

  const renderForgotPasswordForm = () => (
    <div className="space-y-4">
      {resetSent ? (
        <div className="text-center py-4">
          <div className="mb-4 text-soccer-green">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium mb-2">Check Your Email</h3>
          <p className="text-gray-600 mb-4">
            We've sent a password reset link to your email address. Please check your inbox.
          </p>
          <Button onClick={() => handleModeSwitch("login")}>
            Return to Login
          </Button>
        </div>
      ) : (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleForgotPassword();
          }}
          className="space-y-4"
        >
          <div className="space-y-2">
            <p className="text-sm text-gray-600 mb-4">
              Enter your email address and we'll send you a link to reset your password.
            </p>
            <Label htmlFor="reset-email">Email</Label>
            <Input
              id="reset-email"
              type="email"
              placeholder="your@email.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <Button
            type="submit"
            className="w-full bg-soccer-green hover:bg-soccer-green/90"
          >
            Send Reset Link
          </Button>

          <div className="text-center mt-4">
            <button
              type="button"
              className="text-soccer-green hover:underline text-sm"
              onClick={() => handleModeSwitch("login")}
            >
              Back to Login
            </button>
          </div>
        </form>
      )}
    </div>
  );

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {mode === "login"
              ? "Login to Your Account"
              : mode === "register"
              ? "Create an Account"
              : "Reset Password"}
          </DialogTitle>
        </DialogHeader>

        {mode === "login" && (
          <LoginForm
            onSuccess={onSuccess}
            onSwitchToRegister={() => handleModeSwitch("register")}
            onForgotPassword={() => handleModeSwitch("forgot-password")}
          />
        )}

        {mode === "register" && (
          <RegisterForm
            onSuccess={onSuccess}
            onSwitchToLogin={() => handleModeSwitch("login")}
          />
        )}

        {mode === "forgot-password" && renderForgotPasswordForm()}
      </DialogContent>
    </Dialog>
  );
}

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { API } from "@/services/api";
