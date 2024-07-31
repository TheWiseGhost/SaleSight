import React, { useEffect, useState } from "react";
import { Avatar } from "@mui/material";
import { useNavigate } from "react-router-dom";

const Profile = () => {
  const token = localStorage.getItem("token");
  const [user, setUserInfo] = useState([]);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const handleNavigation = (path) => {
    navigate(path);
  };

  useEffect(() => {
    if (!token) {
      navigate("/auth");
    }
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true); // Set loading to true before starting the fetch
      const token = localStorage.getItem("token"); // Retrieve the token from localStorage

      try {
        const response = await fetch("/api/user_info/", {
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
        setUserInfo(data.user[0]);
      } catch (error) {
        console.error("Error fetching user info:", error);
      } finally {
        setLoading(false); // Ensure loading is set to false after fetch attempt
      }
    };

    fetchData();
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/auth");
  };

  return (
    <>
      {loading ? (
        <div className="flex w-3/5 mx-auto pt-60 items-center justify-center">
          <svg class="pl" width="240" height="240" viewBox="0 0 240 240">
            <circle
              class="pl__ring pl__ring--a"
              cx="120"
              cy="120"
              r="105"
              fill="none"
              stroke="#000"
              stroke-width="20"
              stroke-dasharray="0 660"
              stroke-dashoffset="-330"
              stroke-linecap="round"
            ></circle>
            <circle
              class="pl__ring pl__ring--b"
              cx="120"
              cy="120"
              r="35"
              fill="none"
              stroke="#000"
              stroke-width="20"
              stroke-dasharray="0 220"
              stroke-dashoffset="-110"
              stroke-linecap="round"
            ></circle>
            <circle
              class="pl__ring pl__ring--c"
              cx="85"
              cy="120"
              r="70"
              fill="none"
              stroke="#000"
              stroke-width="20"
              stroke-dasharray="0 440"
              stroke-linecap="round"
            ></circle>
            <circle
              class="pl__ring pl__ring--d"
              cx="155"
              cy="120"
              r="70"
              fill="none"
              stroke="#000"
              stroke-width="20"
              stroke-dasharray="0 440"
              stroke-linecap="round"
            ></circle>
          </svg>
        </div>
      ) : (
        <>
          <div className="flex flex-col items-center mt-12">
            <Avatar sx={{ width: 128, height: 128, mb: 2, mt: 8 }} />
            <p className="text-4xl pb-6 font-abo-one">{user.name}</p>
            <p className="text-2xl font-reem-kufi text-gray-700">
              Currently on the {user.plan} plan
            </p>
            <div className="flex flex-row items-center space-x-12 pt-12">
              <button
                onClick={() => handleNavigation("/dashboard")}
                className="px-9 py-4 bg-mygreen hover:bg-lightgreen transition duration-200 text-black border-none rounded-2xl text-xl font-reem-kufi cursor-pointer"
              >
                Go Back
              </button>
              <button
                onClick={() => handleLogout()}
                className="px-9 py-4 bg-yellow-300 hover:bg-yellow-200 transition duration-200 text-black border-none rounded-2xl text-xl font-reem-kufi cursor-pointer"
              >
                Logout
              </button>
            </div>
          </div>
        </>
      )}
    </>
  );
};

export default Profile;
