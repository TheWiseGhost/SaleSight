import React, { useState, useEffect } from "react";
import "../../static/css/index.css";

const PredictionForm = () => {
  const [formData, setFormData] = useState({
    response: "",
    location: "",
    lead_gen: "",
    offer: "",
    sale_product: "",
    funnel_stage: "",
    model: "",
  });

  const [dropdownOptions, setDropdownOptions] = useState({
    responses: [],
    locations: [],
    lead_Gens: [],
    offers: [],
    sale_Products: [],
    funnel_Stages: [],
    models: [],
  });

  const translator = {
    response: "Responses",
    location: "Locations",
    lead_gen: "Lead_Gens",
    offer: "Offers",
    sale_product: "Sale_Products",
    funnel_stage: "Funnel_Stages",
    model: "models",
  };

  const token = localStorage.getItem("token");

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true); // Track loading state

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true); // Set loading to true before starting the fetch
      const token = localStorage.getItem("token"); // Retrieve the token from localStorage

      try {
        const response = await fetch("/api/get_prediction_form/", {
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

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);

    for (let key in formData) {
      if (!formData[key]) {
        setError(`The field "${key}" is required.`);
        return;
      }
    }

    console.log("Submitting form data:", formData);

    try {
      const requestOptions = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`, // Include JWT in headers
        },
        body: JSON.stringify(formData),
      };

      const response = await fetch("/api/prediction/", requestOptions);
      const data = await response.json();

      if (data.auth) {
        navigate("/auth");
      }

      if (data.result) {
        setResult(
          "Model is " +
            (data.prob * 100).toFixed(2) +
            "% sure that " +
            data.result +
            " will occur"
        );
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
      console.log(parsedOptions);
      return parsedOptions.map((option) => (
        <option
          key={option.main}
          value={option.main}
          className="hover:bg-cyan-600"
        >
          {option.main}
        </option>
      ));
    } catch (error) {
      console.error("Error parsing options:", error);
      return null;
    }
  };

  const renderFormFields = () => {
    const formFields = [];
    for (let key in formData) {
      const isModel = ["model"].includes(key);
      formFields.push(
        <div key={key}>
          {isModel ? (
            <div className="flex flex-row my-4">
              <div className="w-1/5">
                <img
                  src="../../static/images/AiIcon.png"
                  alt="Data Portal"
                  className="w-2/3 pt-2"
                />
              </div>
              <select
                id={`form-${key}`}
                name={key}
                value={formData[key]}
                onChange={handleChange}
                className="form-select font-reem-kufi mt-1 block w-full px-3 py-3 rounded-2xl border-cyan-300 border-2 shadow-sm focus:border-cyan-500"
                title={`Select ${key}`}
              >
                <option value="">Select {key}</option>
                {renderDropdownOptions(
                  JSON.stringify(dropdownOptions[translator[key]] || [])
                )}
              </select>
            </div>
          ) : (
            <>
              <label
                htmlFor={`form-${key}`}
                className="block text-md pl-4 font-abo-one text-gray-800"
              >
                {key.charAt(0).toUpperCase() + key.slice(1)}:
              </label>
              <select
                id={`form-${key}`}
                name={key}
                value={formData[key]}
                onChange={handleChange}
                className="form-select font-reem-kufi mt-1 block w-full px-3 py-3 rounded-2xl border-cyan-300 border-2 shadow-sm focus:border-cyan-500"
                title={`Select ${key}`}
              >
                <option value="">Select {key}</option>
                {renderDropdownOptions(
                  JSON.stringify(dropdownOptions[translator[key]] || [])
                )}
              </select>
            </>
          )}
        </div>
      );
    }
    return formFields;
  };

  // Render loading state while fetching data
  if (loading) {
    return (
      <div>
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
      </div>
    );
  }

  // Render form once data is fetched
  return (
    <div className="max-w-lg mx-auto p-6 bg-white rounded-md">
      <h2 className="text-4xl text-cyan-600 font-abo-one mb-8">
        Get Prediction
      </h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        {renderFormFields()}
        <div className="w-1/2 mx-auto pt-4">
          <button
            type="submit"
            className="w-full font-reem-kufi text-lg text-black py-3 px-4 rounded-2xl bg-cyan-300 focus:outline-none hover:bg-cyan-400"
          >
            Predict
          </button>
        </div>
      </form>
      {result && (
        <div className="mt-4 w-full overflow-auto">
          <h3 className="text-xl font-bold">Result:</h3>
          <pre className="bg-gray-100 p-4 rounded-md">{result}</pre>
        </div>
      )}
      {error && (
        <div className="mt-4 text-red-500">
          <h3 className="text-xl font-bold">Error:</h3>
          <pre className="bg-gray-100 p-4 rounded-md">{error}</pre>
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
