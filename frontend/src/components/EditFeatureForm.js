import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../../static/css/index.css";

const EditFeatureForm = () => {
  const [activeFeature, setActiveFeature] = useState("Offer");
  const [activeOption, setActiveOption] = useState("");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const token = localStorage.getItem("token"); // Retrieve the token from localStorage

  const translator = {
    Response: "Responses",
    Location: "Locations",
    Lead: "Lead_Gens",
    Offer: "Offers",
    Product: "Sale_Products",
    "Funnel Stage": "Funnel_Stages",
    Result: "Results",
  };

  const [dropdownOptions, setDropdownOptions] = useState({
    Responses: [],
    Locations: [],
    Lead_Gens: [],
    Offers: [],
    Sale_Products: [],
    Funnel_Stages: [],
    Results: [],
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true); // Set loading to true before starting the fetch
      const token = localStorage.getItem("token"); // Retrieve the token from localStorage

      try {
        const response = await fetch("/api/get_feature_values/", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`, // Include JWT in headers
          },
        });
        if (response.auth) {
          navigate("/auth");
        }
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        console.log("Fetched dropdown options:", data);
        setDropdownOptions(data);
      } catch (error) {
        console.error("Error fetching dropdown options:", error);
        setError("Error fetching dropdown options");
      } finally {
        setLoading(false); // Ensure loading is set to false after fetch attempt
      }
    };

    fetchData();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch("/api/edit_feature/", {
        method: "POST",
        body: JSON.stringify({
          feature: translator[activeFeature],
          main: activeOption,
          text: text,
        }),
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await response.json();

      if (data.success) {
        setResult(data.success);
        setError(null);
      } else {
        setError(data.error || "Unknown error occurred");
        setResult(null);
      }
    } catch (error) {
      console.error("Error submitting data:", error);
      setError("Error submitting data");
    }
  };

  const renderDropdownOptions = (options) => {
    try {
      const parsedOptions = JSON.parse(options);
      return (
        <select
          value={activeOption}
          onChange={(e) => setActiveOption(e.target.value)}
          className="border-2 block w-full px-3 py-2 rounded-2xl border-gray-300 shadow-sm focus:border-gray-400"
        >
          <option>Select option</option>
          {parsedOptions.map((option) => (
            <option key={option.main} value={option.main}>
              {option.main}
            </option>
          ))}
        </select>
      );
    } catch (error) {
      console.error("Error parsing options:", error);
      return null;
    }
  };

  // Render form once data is fetched
  return (
    <div className="max-w-lg mx-auto p-6 bg-white rounded-md">
      <h2 className="text-4xl font-abo-one text-gray-700 pb-6">Edit Feature</h2>
      {loading ? (
        <div className="flex w-full pt-12 items-center justify-center">
          <svg className="pl" width="240" height="240" viewBox="0 0 240 240">
            <circle
              className="pl__ring pl__ring--a"
              cx="120"
              cy="120"
              r="105"
              fill="none"
              stroke="#000"
              strokeWidth="20"
              strokeDasharray="0 660"
              strokeDashoffset="-330"
              strokeLinecap="round"
            ></circle>
            <circle
              className="pl__ring pl__ring--b"
              cx="120"
              cy="120"
              r="35"
              fill="none"
              stroke="#000"
              strokeWidth="20"
              strokeDasharray="0 220"
              strokeDashoffset="-110"
              strokeLinecap="round"
            ></circle>
            <circle
              className="pl__ring pl__ring--c"
              cx="85"
              cy="120"
              r="70"
              fill="none"
              stroke="#000"
              strokeWidth="20"
              strokeDasharray="0 440"
              strokeLinecap="round"
            ></circle>
            <circle
              className="pl__ring pl__ring--d"
              cx="155"
              cy="120"
              r="70"
              fill="none"
              stroke="#000"
              strokeWidth="20"
              strokeDasharray="0 440"
              strokeLinecap="round"
            ></circle>
          </svg>
        </div>
      ) : (
        <>
          <form onSubmit={handleSubmit} className="space-y-6">
            <select
              onChange={(e) => setActiveFeature(e.target.value)}
              className="border-2 block w-full px-3 py-2 rounded-2xl border-gray-300 shadow-sm focus:border-gray-400"
            >
              {[
                "Offer",
                "Product",
                "Location",
                "Lead",
                "Funnel Stage",
                "Response",
                "Result",
              ].map((buttonName) => (
                <option key={buttonName} className="py-2 px-4 w-full">
                  {buttonName}
                </option>
              ))}
            </select>

            {renderDropdownOptions(
              JSON.stringify(dropdownOptions[translator[activeFeature]] || [])
            )}

            <input
              type="text"
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter a New Name"
              className="w-full p-3 border-gray-300 border-2 rounded-2xl font-reem-kufi"
            />
            <div className="w-4/5 mx-auto pt-4">
              <button
                type="submit"
                className="w-full font-reem-kufi text-lg text-black py-3 px-4 rounded-2xl bg-gray-200 focus:outline-none hover:bg-gray-300"
              >
                Update Feature
              </button>
            </div>
          </form>
          {result && (
            <div className="mt-4 text-green-500">
              <h3 className="text-xl font-bold">Result:</h3>
              <pre className="bg-gray-100 p-4 rounded-md">
                {JSON.stringify("Successfully edited row", null, 2)}
              </pre>
            </div>
          )}
          {error && (
            <div className="mt-4 text-gray-500">
              <h3 className="text-xl font-bold">Error:</h3>
              <pre className="bg-gray-100 p-4 rounded-md">{error}</pre>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default EditFeatureForm;
