import React from "react";

const AiExample = () => {
  return (
    <div className="flex flex-col items-center justify-center p-6 pb-16">
      <h2 className="text-4xl text-center md:text-left italic mb-12 font-abo-one">
        “Love how I get the exact numbers I need to know”
      </h2>
      <div className="flex flex-col md:flex-row items-center justify-center space-x-6">
        <div className="flex flex-row items-center justify-center border-4 border-gray-600 rounded-lg p-4 md:w-1/3">
          <div className="p-2">
            <img src="../../static/images/AiIcon.png" className="w-full" />
          </div>
          <p className="text-center font-reem-kufi text-xl">
            “Consider moving more resources to generating leads from Facebook”
          </p>
        </div>
        <span className="text-5xl">→</span>
        <div className="flex flex-col items-center font-reem-kufi justify-center border-4 border-blue-600 rounded-lg p-4 mb-4 md:mb-0 md:w-1/5">
          <p className="text-center text-lg font-semibold ">
            Lead - Facebook Group
          </p>
          <p className="text-center text-2xl my-2">8.94%</p>
          <p className="text-center">Result in “Bought Offer”</p>
        </div>
        <div className="flex flex-col items-center font-reem-kufi justify-center border-4 border-blue-400 rounded-lg p-4 md:w-1/5">
          <p className="text-center text-lg font-semibold ">Lead - TikTok</p>
          <p className="text-center text-2xl my-2">2.54%</p>
          <p className="text-center">Result in “Bought Offer”</p>
        </div>
      </div>
    </div>
  );
};

export default AiExample;
