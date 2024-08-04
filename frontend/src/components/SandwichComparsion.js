import React from "react";
import { useNavigate } from "react-router-dom";

const SandwichComparsion = () => {
  const navigate = useNavigate();
  const handleNavigation = (path) => {
    navigate(path);
  };
  return (
    <div className="flex flex-col justify-center items-center min-h-screen p-6 pb-20">
      <div className="text-center mb-6">
        <h2 className="text-3xl font-reem-kufi">
          For <span className="text-brightgreen">$9.99</span> you get...
        </h2>
      </div>
      <div className="flex flex-col md:flex-row justify-center items-start md:space-x-10 mb-6">
        <div className="bg-white p-6 pt-4 rounded-md max-w-xs md:max-w-none">
          <h3 className="text-center text-2xl font-abo-one mb-6">Sale Sight</h3>
          <div className="w-4/5 ml-auto">
            <ul className="list-disc space-y-4 font-reem-kufi list-outside text-lg">
              <li>Unlimited AI Predictions</li>
              <li>Unlimited AI Models</li>
              <li>Free Data Storage</li>
              <li>Saves endless time from chasing the same dead lead types</li>
              <li>Key Data Insights for adjustments</li>
              <li>Grows your Business</li>
              <li>Makes use of all your spreadsheet data</li>
            </ul>
          </div>
        </div>
        <div className="bg-white p-6 pt-4 rounded-md max-w-xs md:max-w-none">
          <h3 className="text-center font-abo-one text-2xl mb-6">Sandwich</h3>
          <div className="w-4/5 ml-auto">
            <ul className="list-disc space-y-4 font-reem-kufi list-outside text-lg">
              <li>Unlimited Water?</li>
              <li>Unlimited Napkins?</li>
              <li>Free mystery meats</li>
              <li>Saves about 600 extra unwanted calories in your body</li>
              <li>Key Time to reflect on life choices</li>
              <li>Grows your belly</li>
              <li>Makes use of all your spare change</li>
            </ul>
          </div>
        </div>
      </div>
      <div className="text-center mb-4">
        <h2 className="text-2xl font-reem-kufi">It's Your Call</h2>
      </div>
      <div className="flex space-x-20">
        <button
          onClick={() => handleNavigation("/chooseplan")}
          className="bg-mygreen font-abo-one hover:bg-lightgreen text-black px-8 py-4 rounded-xl transition duration-200"
        >
          Sale Sight
        </button>
        <button
          onClick={() => {
            window.open("https://www.jimmyjohns.com/", "_blank");
          }}
          className="bg-yellow-300 font-abo-one hover:bg-amber-200 text-black px-8 py-4 rounded-xl transition duration-200"
        >
          Sandwich
        </button>
      </div>
      <div className="text-center mt-28 flex flex-col space-y-8">
        <h2 className="text-5xl font-abo-one">Still Unsure?</h2>
        <button
          onClick={() => handleNavigation("/auth")}
          className="bg-cyan-300 w-3/5 mx-auto text-xl font-reem-kufi hover:bg-cyan-400 transition duration-200 text-black px-5 py-4 rounded-2xl"
        >
          Try for Free
        </button>
      </div>
    </div>
  );
};

export default SandwichComparsion;
