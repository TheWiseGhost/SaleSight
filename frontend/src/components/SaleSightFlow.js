import React from "react";
import Lottie from "react-lottie";
import flow from "../../static/animations/SaleSightFlow.json";
import { useMediaQuery } from "react-responsive";

const SaleSightFlow = () => {
  const isSmallScreen = useMediaQuery({ maxWidth: 768 });
  const flowOptions = {
    loop: true,
    autoplay: true,
    animationData: flow,
    rendererSettings: {
      preserveAspectRatio: "xMidYMid slice",
    },
  };
  return (
    <div className="h-80 pt-12">
      <div className="flex flex-row">
        <div className="hidden md:flex w-1/2 pl-12 justify-start text-center flex-col">
          <h1 className="text-5xl font-abo-one leading-tight text-gray-800">
            <span className="text-brightgreen">Track</span> your Data if you
            want to <span className="text-brightgreen">Get Ahead</span>
          </h1>
          <h4 className="text-xl text-gray-700 font-reem-kufi mt-12">
            "If you don't track, you don't care" -{" "}
            <span className="text-black">Alex Hormozi</span>
          </h4>
          <h4 className="text-xl text-gray-700 font-reem-kufi mt-6 px-16">
            “Without big data, you are blind and deaf and in the middle of a
            freeway” — <span className="text-black">Geoffrey Moore</span>
          </h4>
        </div>
        {isSmallScreen ? (
          <div className="w-1/2 justify-center">
            <Lottie options={flowOptions} width={470} height={280} />
          </div>
        ) : (
          <div className="w-1/2 justify-start pr-12">
            <Lottie options={flowOptions} width={500} height={300} />
          </div>
        )}
      </div>
    </div>
  );
};

export default SaleSightFlow;
