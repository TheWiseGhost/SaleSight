import React, { useEffect, useState } from "react";
import IconButton from "@mui/material/IconButton";
import DeleteIcon from "@mui/icons-material/Delete";
import { useNavigate } from "react-router-dom";

const DeleteFeatureForm = () => {
  const [features, setFeatures] = useState({});
  const [activeFeature, setActiveFeature] = useState("Offer");
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

  const fetchFeatures = async () => {
    try {
      setLoading(true);
      const response = await fetch("/api/get_feature_values/", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`, // Include JWT in headers
        },
      });
      const data = await response.json();
      if (data.auth) {
        navigate("/auth");
      }
      console.log(data);
      setFeatures(data || {});
      setLoading(false);
    } catch (error) {
      console.error("Error fetching features:", error);
    }
  };

  useEffect(() => {
    fetchFeatures();
  }, []);

  const handleDeleteFeature = async (feature, option) => {
    try {
      const formData = new FormData();
      formData.append("feature", feature);
      formData.append("option", option);

      const response = await fetch("/api/delete_feature/", {
        method: "POST",
        body: formData,
        headers: {
          Authorization: `Bearer ${token}`, // Include JWT in headers
        },
      });

      if (response.auth) {
        navigate("/auth");
      }

      if (response.ok) {
        window.location.reload();
      } else {
        console.error("Failed to delete feature");
      }
    } catch (error) {
      console.error("Error deleting feature:", error);
    }
  };

  const renderFeatures = (options) => {
    try {
      if (options != "[]") {
        const parsedOptions = JSON.parse(options);
        return parsedOptions.map((option) => (
          <div
            className="flex justify-between items-center font-reem-kufi pb-8"
            key={option.main}
          >
            {option.main}
            <IconButton
              onClick={() =>
                handleDeleteFeature(translator[activeFeature], option.main)
              }
              style={{
                backgroundColor: "#FF6C63",
                color: "white",
                width: "2rem",
                height: "2rem",
              }}
            >
              <DeleteIcon fontSize="inherit" />
            </IconButton>
          </div>
        ));
      } else {
        return (
          <div className="font-reem-kufi">No {activeFeature} available</div>
        );
      }
    } catch (error) {
      console.error("Error parsing options:", error);
      return null;
    }
  };

  return (
    <div className="max-w-lg mx-auto p-6 bg-white rounded-md">
      <h2 className="text-4xl text-red-500 font-abo-one pb-8">
        Delete Feature
      </h2>
      <div className="mt-6 space-y-4 text-lg text-center font-reem-kufi">
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
            <select
              onChange={(e) => setActiveFeature(e.target.value)}
              className="border-2 block w-full px-3 py-2 rounded-2xl border-red-300 shadow-sm focus:border-red-400"
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
            <div className="pt-6 h-fit">
              {renderFeatures(
                JSON.stringify(features[translator[activeFeature]] || [])
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default DeleteFeatureForm;
