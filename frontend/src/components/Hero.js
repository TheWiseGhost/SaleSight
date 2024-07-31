import React from "react";
import { useNavigate } from "react-router-dom";

const Hero = ({ scrollToSection }) => {
  const navigate = useNavigate();

  const handleNavigation = (path) => {
    navigate(path);
  };

  const handleClick = (path) => {
    scrollToSection(path);
  };

  return (
    <div className="min-h-screen pb-20 flex flex-col">
      {/* Navbar */}
      <div className="text-black font-reem-kufi py-4 px-6 mx-6 mt-6 flex justify-between items-center">
        <div className="flex flex-row items-center">
          <img
            src="../../static/images/SaleSightLogo.png"
            className="w-24 items-center pl-4 pr-6"
          />
          <div className="font-abo-one hidden md:block text-2xl">
            Sale Sight
          </div>
        </div>
        <div className="space-x-12 flex flex-row text-lg md:mr-8">
          {/* Navbar buttons */}
          <div className="hidden space-x-12 md:flex flex-row">
            <button
              onClick={() => handleClick("demo")}
              className="text_button font-reem-kufi bg-white text-gray-700"
            >
              <span className="span-mother">
                <span>D</span>
                <span>e</span>
                <span>m</span>
                <span>o </span>
              </span>
              <span className="span-mother2">
                <span>D</span>
                <span>e</span>
                <span>m</span>
                <span>o </span>
              </span>
            </button>
            <button
              onClick={() => handleClick("reviews")}
              className="text_button font-reem-kufi bg-white text-gray-700"
            >
              <span className="span-mother">
                <span>R</span>
                <span>e</span>
                <span>v</span>
                <span>i </span>
                <span>e</span>
                <span>w</span>
                <span>s</span>
              </span>
              <span className="span-mother2">
                <span>R</span>
                <span>e</span>
                <span>v</span>
                <span>i </span>
                <span>e</span>
                <span>w</span>
                <span>s</span>
              </span>
            </button>
            <button
              onClick={() => handleClick("example")}
              className="text_button font-reem-kufi bg-white text-gray-700"
            >
              <span className="span-mother">
                <span>E</span>
                <span>x</span>
                <span>a</span>
                <span>m </span>
                <span>p</span>
                <span>l</span>
                <span>e</span>
              </span>
              <span className="span-mother2">
                <span>E</span>
                <span>x</span>
                <span>a</span>
                <span>m </span>
                <span>p</span>
                <span>l</span>
                <span>e</span>
              </span>
            </button>
          </div>
          <div className="flex flex-row space-x-6 pl-8">
            <button
              onClick={() => handleNavigation("/auth")}
              className="bg-blue-300 font-reem-kufi hover:bg-blue-400 transition duration-200 text-black my-2 md:my-0 px-5 py-2 rounded-xl"
            >
              Login
            </button>
            <button
              onClick={() => handleNavigation("/auth")}
              className="bg-cyan-300 font-reem-kufi hover:bg-cyan-400 transition duration-200 text-black px-5 py-2 rounded-xl"
            >
              Try for Free
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 flex flex-col items-center">
        {/* Title */}
        <h1 className="text-5xl md:text-7xl w-4/5 md:w-3/5 text-center leading-normal md:leading-snug mx-auto font-abo-one mt-8">
          Get AI Insights for the Price of a Sandwich
        </h1>
        {/* Subtitle */}
        <p className="text-2xl mt-8 text-center font-reem-kufi text-neutral-500">
          Then use it to make enough to buy thousands of sandwiches!!!
        </p>
        {/* Buttons row */}
        <div className="mt-12 space-x-16 flex flex-row">
          <button
            onClick={() => handleNavigation("/chooseplan")}
            className="px-9 py-4 bg-mygreen hover:bg-lightgreen transition duration-200 text-black border-none rounded-2xl text-xl font-reem-kufi cursor-pointer"
          >
            Let's Go
          </button>
          <button
            onClick={() => handleClick("demo")}
            className="px-5 py-4 bg-yellow-300 hover:bg-yellow-200 transition duration-200 text-black border-none rounded-2xl text-xl font-reem-kufi cursor-pointer"
          >
            Learn More
          </button>
        </div>
        {/* Image */}
        <div className="mt-8">
          <img
            src="../../static/images/SaleSightCurve.png"
            alt="Description"
            className="w-screen h-auto"
          />
        </div>

        {/* Link with arrow */}
        <div className="arrow_container *:hidden md:flex  justify-center mt-12 w-1/3 mx-auto">
          <button className="arrow_button">
            Still think the sandwich is better?
          </button>
          <span className="arrow first">
            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path d="M16.1716 10.9999L10.8076 5.63589L12.2218 4.22168L20 11.9999L12.2218 19.778L10.8076 18.3638L16.1716 12.9999H4V10.9999H16.1716Z"></path>
            </svg>
          </span>
          <span className="arrow second">
            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path d="M16.1716 10.9999L10.8076 5.63589L12.2218 4.22168L20 11.9999L12.2218 19.778L10.8076 18.3638L16.1716 12.9999H4V10.9999H16.1716Z"></path>
            </svg>
          </span>
          <span className="arrow third">
            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path d="M16.1716 10.9999L10.8076 5.63589L12.2218 4.22168L20 11.9999L12.2218 19.778L10.8076 18.3638L16.1716 12.9999H4V10.9999H16.1716Z"></path>
            </svg>
          </span>
        </div>
      </div>
    </div>
  );
};

export default Hero;
