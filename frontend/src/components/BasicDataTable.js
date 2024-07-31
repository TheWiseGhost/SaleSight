import React from "react";
import Lottie from "react-lottie";
import animationData from "../../static/animations/SimpleDataTable.json";

const BasicDataTable = () => {
  const defaultOptions = {
    loop: true,
    autoplay: true,
    animationData: animationData,
    rendererSettings: {
      preserveAspectRatio: "xMidYMid slice",
    },
  };

  return (
    <div className="flex flex-col md:flex-row h-screen pb-20">
      <div className="flex-1 flex items-center justify-center bg-white">
        <h1 className="text-3xl md:text-7xl text-center text-gray-900 md:text-left px-4 md:px-0 font-abo-one md:pl-20 leading-snug">
          Real Time <span className="text-cyan-500">AI Predictions</span> -
          Using Your Data
        </h1>
      </div>
      <div className="flex-1 flex pr-0 md:pt-8 items-center justify-center bg-white">
        <Lottie options={defaultOptions} height={400} width={400} />
      </div>
    </div>
  );
};

export default BasicDataTable;
