import React from "react";
import Lottie from "react-lottie";
import aiIcon from "../../static/animations/AiIcon.json";
import dataIcon from "../../static/animations/DataIcon.json";
import { useNavigate } from "react-router-dom";

const SimpleCTA = ({ scrollToSection }) => {
  const navigate = useNavigate();
  const handleNavigation = (path) => {
    navigate(path);
  };

  const handleClick = (path) => {
    scrollToSection(path);
  };

  const dataIconOptions = {
    loop: true,
    autoplay: true,
    animationData: dataIcon,
    rendererSettings: {
      preserveAspectRatio: "xMidYMid slice",
    },
  };
  const aiIconOptions = {
    loop: true,
    autoplay: true,
    animationData: aiIcon,
    rendererSettings: {
      preserveAspectRatio: "xMidYMid slice",
    },
  };
  return (
    <div className="flex flex-col justify-center h-screen space-y-20 pt-4 pb-12">
      <div className="flex flex-col items-center space-y-24 ">
        <div className="flex justify-center items-center space-x-8 md:space-x-12 md:pr-12">
          <div className="hidden md:flex">
            <Lottie options={aiIconOptions} height={200} width={200} />
          </div>
          <div className="flex md:hidden">
            <Lottie options={aiIconOptions} height={100} width={100} />
          </div>
          <div className="flex flex-col items-center">
            <h1 className="text-4xl font-abo-one">
              Create <span className="text-blue-400">Your Own</span> AI Models
            </h1>
            <button
              onClick={() => handleNavigation("/auth")}
              className="mt-6 px-8 py-4 font-reem-kufi bg-blue-300 hover:bg-blue-400 transition duration-200 text-black rounded-xl"
            >
              Let's Go
            </button>
          </div>
        </div>
        <div className="flex justify-center items-center">
          <div className="flex flex-col items-center pr-6 md:pr-12">
            <h1 className="text-4xl font-abo-one pl-6 md:pl-0">
              Store <span className="text-yellow-500">All</span> of Your Data
              Easily
            </h1>
            <button
              onClick={() => handleClick("example")}
              className="mt-6 px-6 py-4 font-reem-kufi bg-amber-300 hover:bg-amber-400 transition duration-200 text-black rounded-xl"
            >
              See Example
            </button>
          </div>
          <div className="hidden md:flex">
            <Lottie options={dataIconOptions} height={200} width={200} />
          </div>
          <div className="pr-6 flex md:hidden">
            <Lottie options={dataIconOptions} height={100} width={100} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default SimpleCTA;
