import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const Auth = () => {
  const [isSignUpActive, setIsSignUpActive] = useState(false);
  const [signUpData, setSignUpData] = useState({
    name: "",
    email: "",
    password: "",
  });
  const [loginData, setLoginData] = useState({
    email: "",
    password: "",
  });
  const navigate = useNavigate();

  const handleNavigation = (path) => {
    navigate(path);
  };

  const handleSignUpClick = () => {
    setIsSignUpActive(true);
  };

  const handleSignInClick = () => {
    setIsSignUpActive(false);
  };

  const handleSignUpChange = (e) => {
    setSignUpData({ ...signUpData, [e.target.name]: e.target.value });
  };

  const handleLoginChange = (e) => {
    setLoginData({ ...loginData, [e.target.name]: e.target.value });
  };

  const handleSignUpSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("/api/sign_up/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(signUpData),
      });
      if (response.ok) {
        alert(
          "Sign-up successful. Please check your email to verify your account"
        );
      } else {
        alert("Sign-up failed");
      }
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("/api/login/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(loginData),
      });
      const data = await response.json();
      if (response.ok) {
        if (data.pay) {
          alert("Your free trial has finsihed. Please choose a payment plan.");
          handleNavigation("/chooseplan");
        } else if (data.verify) {
          alert("Check your email to verify your account");
        } else {
          localStorage.setItem("token", data.access_token);
          // localStorage.setItem("refresh_token", data.refresh_token);
          handleNavigation("/dashboard");
        }
      } else {
        alert("Login failed");
      }
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className="h-screen pt-12">
      <div
        className={`container mx-auto ${
          isSignUpActive ? "right-panel-active" : ""
        }`}
        id="container"
      >
        <div className="form-container sign-up-container">
          <form
            className="bg-white flex items-center justify-center flex-col px-4 md:px-12 h-full text-center"
            onSubmit={handleSignUpSubmit}
          >
            <img
              src="../../static/images/SaleSightLogo.png"
              className="w-24 items-center pl-4 pr-6 pb-4"
            />
            <h1 className="text-3xl font-abo-one mb-4">Create Account</h1>
            <input
              className="auth_input font-reem-kufi"
              type="text"
              name="name"
              placeholder="Name"
              value={signUpData.name}
              onChange={handleSignUpChange}
            />
            <input
              className="auth_input font-reem-kufi"
              type="email"
              name="email"
              placeholder="Email"
              value={signUpData.email}
              onChange={handleSignUpChange}
            />
            <input
              className="auth_input font-reem-kufi"
              type="password"
              name="password"
              placeholder="Password"
              value={signUpData.password}
              onChange={handleSignUpChange}
            />
            <button
              className="font-abo-one mt-4 text-lg auth_button bg-blue-300 transition duration-300 hover:bg-blue-400 py-3 px-6 rounded-2xl"
              type="submit"
            >
              Sign Up
            </button>
          </form>
        </div>
        <div className="form-container sign-in-container">
          <form
            className="bg-white flex items-center justify-center flex-col px-4 md:px-12 h-full text-center"
            onSubmit={handleLoginSubmit}
          >
            <img
              src="../../static/images/SaleSightLogo.png"
              className="w-24 items-center pl-4 pr-6 pb-4"
            />
            <h1 className="text-3xl font-abo-one mb-4">Sign in</h1>
            <input
              className="auth_input font-reem-kufi"
              type="email"
              name="email"
              placeholder="Email"
              value={loginData.email}
              onChange={handleLoginChange}
            />
            <input
              className="auth_input font-reem-kufi"
              type="password"
              name="password"
              placeholder="Password"
              value={loginData.password}
              onChange={handleLoginChange}
            />
            <a className="text-sm text-gray-600 mb-2 hover:underline" href="#">
              Forgot your password?
            </a>
            <button
              className="font-abo-one mt-4 text-lg auth_button bg-blue-300 transition duration-300 hover:bg-blue-400 py-3 px-6 rounded-2xl"
              type="submit"
            >
              Login
            </button>
          </form>
        </div>
        <div className="overlay-container">
          <div className="overlay">
            <div className="overlay-panel overlay-left">
              <h1 className="text-3xl font-reem-kufi mb-4">
                Already have an account?
              </h1>
              <p className="font-reem-kufi text-xl mb-8">
                Welcome back! Click down below to log back into Sale Sight
              </p>
              <button
                className="auth_button_ghost font-abo-one px-6 py-3 text-xl rounded-2xl"
                id="signIn"
                onClick={handleSignInClick}
              >
                Login
              </button>
            </div>

            <div className="overlay-panel overlay-right">
              <h1 className="text-3xl font-reem-kufi mb-4 ">
                Don't have an account?
              </h1>
              <p className="font-reem-kufi text-xl mb-8">
                Click down below to get a 14 Day free trial
              </p>
              <button
                className="auth_button_ghost font-abo-one px-6 py-3 text-xl rounded-2xl"
                id="signUp"
                onClick={handleSignUpClick}
              >
                Sign Up
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Auth;
