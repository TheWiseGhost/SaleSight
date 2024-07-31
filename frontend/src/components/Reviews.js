"use client";

import React from "react";

const Reviews = () => {
  const handleMouseMove = (e, id) => {
    return;
  };

  const handleMouseLeave = (id) => {
    const card = document.getElementById(`${id}`);
    card.style.transform = `rotateY(0deg) rotateX(0deg)`;
  };

  return (
    <div className="md:min-h-screen pt-12 pb-16" id="reviews">
      <h1 className="font-abo-one text-5xl text-center pb-10">
        Don't Take Just Our Word for It!
      </h1>
      <div className="hidden md:grid grid-cols-4 grid-rows-3 gap-8 py-5 px-12 bg-white h-screen">
        {/* First column */}
        <div className="review-card border-4 transition duration-200 flex flex-col bg-white border-slate-400 col-span-1 row-span-2 rounded-xl">
          <div
            className="card-inner"
            id="card-1"
            onMouseMove={(e) => handleMouseMove(e, "card-1")}
            onMouseLeave={() => handleMouseLeave("card-1")}
          >
            <div className="card-back flex items-center justify-center">
              <h2 className="text-yellow-300 font-abo-one text-3xl">???</h2>
            </div>
            <div className="card-front">
              <div className="pt-4 pb-2">
                <h2 className="text-center text-black font-abo-one text-md mt-2">
                  Jaiti Viyda
                </h2>
                <h1 className="text-center text-yellow-400 text-3xl">★★★★★</h1>
              </div>
              <div className="text-slate-800 font-reem-kufi text-lg px-6 text-center leading-loose">
                "It's hard to say no when they give you such a good offer. I'm
                really glad I skipped out on the sandwich XD! Seriously though,
                I wish I started tracking my data sooner!"
              </div>
            </div>
          </div>
        </div>
        <div className="review-card border-4 transition duration-200 flex flex-col bg-white border-slate-400 col-span-1 row-span-1 row-start-3 rounded-xl">
          <div
            className="card-inner"
            id="card-2"
            onMouseMove={(e) => handleMouseMove(e, "card-2")}
            onMouseLeave={() => handleMouseLeave("card-2")}
          >
            <div className="card-back flex items-center justify-center">
              <h2 className="text-red-400 font-abo-one text-3xl">???</h2>
            </div>
            <div className="card-front">
              <div className="pt-4 pb-2">
                <h2 className="text-center text-black font-abo-one text-md mt-2">
                  J.J. Marshal
                </h2>
                <h1 className="text-center text-yellow-400 text-3xl">★★★★★</h1>
              </div>
              <div className="text-slate-800 font-reem-kufi text-lg px-8 text-center">
                "Amazing customer service!"
              </div>
            </div>
          </div>
        </div>

        {/* Second column */}
        <div className="review-card border-4 transition duration-200 flex flex-col bg-white border-slate-400 col-span-1 row-span-1 rounded-xl">
          <div
            className="card-inner"
            id="card-3"
            onMouseMove={(e) => handleMouseMove(e, "card-3")}
            onMouseLeave={() => handleMouseLeave("card-3")}
          >
            <div className="card-back flex items-center justify-center">
              <h2 className="text-mygreen font-abo-one text-3xl">???</h2>
            </div>
            <div className="card-front">
              <div className="pt-4 pb-2">
                <h2 className="text-center text-black font-abo-one text-md mt-2">
                  Markel Brown
                </h2>
                <h1 className="text-center text-yellow-400 text-3xl">★★★★★</h1>
              </div>
              <div className="text-slate-800 font-reem-kufi text-xl px-6 text-center">
                "Has definitly helped me become more efficient."
              </div>
            </div>
          </div>
        </div>
        <div className="review-card border-4 transition duration-200 flex flex-col bg-white border-slate-400 col-span-1 row-span-2 row-start-2 rounded-xl">
          <div
            className="card-inner"
            id="card-4"
            onMouseMove={(e) => handleMouseMove(e, "card-4")}
            onMouseLeave={() => handleMouseLeave("card-4")}
          >
            <div className="card-back flex items-center justify-center">
              <h2 className="text-cyan-400 font-abo-one text-3xl">???</h2>
            </div>
            <div className="card-front">
              <div className="pt-4 pb-2">
                <h2 className="text-center text-black font-abo-one text-md mt-2">
                  Lisa Wagnor
                </h2>
                <h1 className="text-center text-yellow-400 text-3xl">★★★★★</h1>
              </div>
              <div className="text-slate-800 font-reem-kufi text-lg px-8 text-center leading-loose">
                "You constantly hear people talk about getting more data but
                this is the first time I've actually seen a quick and cheap way
                for getting results"
              </div>
            </div>
          </div>
        </div>

        {/* Third and Fourth columns */}
        <div className="review-card border-4 transition duration-200 flex flex-col bg-white border-slate-400 col-span-2 row-span-1 row-start-3 col-start-3 rounded-xl">
          <div
            className="card-inner"
            id="card-5"
            onMouseMove={(e) => handleMouseMove(e, "card-5")}
            onMouseLeave={() => handleMouseLeave("card-5")}
          >
            <div className="card-back flex items-center justify-center">
              <h2 className="text-yellow-300 font-abo-one text-3xl">???</h2>
            </div>
            <div className="card-front">
              <div className="pt-4 pb-2">
                <h2 className="text-center text-black font-abo-one text-md mt-2">
                  Chris Sarparsis
                </h2>
                <h1 className="text-center text-yellow-400 text-3xl">★★★★★</h1>
              </div>
              <div className="text-slate-800 font-reem-kufi text-lg px-8 text-center">
                "I bought this thinking - 'Hey, at worst I skip lunch' - but it
                truly has changed the way I do outreach"
              </div>
            </div>
          </div>
        </div>
        <div className="review-card border-4 transition duration-200 flex flex-col bg-white border-slate-400 col-span-1 row-span-2 row-start-1 col-start-3 rounded-xl">
          <div
            className="card-inner"
            id="card-6"
            onMouseMove={(e) => handleMouseMove(e, "card-6")}
            onMouseLeave={() => handleMouseLeave("card-6")}
          >
            <div className="card-back flex items-center justify-center">
              <h2 className="text-red-400 font-abo-one text-3xl">???</h2>
            </div>
            <div className="card-front">
              <div className="pt-4 pb-2">
                <h2 className="text-center text-black font-abo-one text-md mt-2">
                  Emily Russo
                </h2>
                <h1 className="text-center text-yellow-400 text-3xl">★★★★★</h1>
              </div>
              <div className="text-slate-800 font-reem-kufi text-lg px-8 text-center leading-loose">
                "Crazy that all it took was logging data to open my eyes! Thanks
                for helping me realize my clients are all coming from my creepy
                Instagram DMs!
              </div>
            </div>
          </div>
        </div>
        <div className="review-card border-4 transition duration-200 flex flex-col bg-white border-slate-400 col-span-1 row-span-1 row-start-1 col-start-4 rounded-xl">
          <div
            className="card-inner"
            id="card-7"
            onMouseMove={(e) => handleMouseMove(e, "card-7")}
            onMouseLeave={() => handleMouseLeave("card-7")}
          >
            <div className="card-back flex items-center justify-center">
              <h2 className="text-yellow-300 font-abo-one text-3xl">???</h2>
            </div>
            <div className="card-front">
              <div className="pt-4 pb-2">
                <h2 className="text-center text-black font-abo-one text-md mt-2">
                  Sophia Zhao
                </h2>
                <h1 className="text-center text-yellow-400 text-3xl">★★★★★</h1>
              </div>
              <div className="text-slate-800 font-reem-kufi text-lg px-6 text-center">
                "I feel like I have a co-founder with my AI!"
              </div>
            </div>
          </div>
        </div>
        <div className="review-card border-4 transition duration-200 flex flex-col bg-white border-slate-400 col-span-1 row-span-1 row-start-2 col-start-4 rounded-xl">
          <div
            className="card-inner"
            id="card-8"
            onMouseMove={(e) => handleMouseMove(e, "card-8")}
            onMouseLeave={() => handleMouseLeave("card-8")}
          >
            <div className="card-back flex items-center justify-center">
              <h2 className="text-cyan-400 font-abo-one text-3xl">???</h2>
            </div>
            <div className="card-front">
              <div className="pt-4 pb-2">
                <h2 className="text-center text-black font-abo-one text-md mt-2">
                  Jose Martinez
                </h2>
                <h1 className="text-center text-yellow-400 text-3xl">★★★★★</h1>
              </div>
              <div className="text-slate-800 font-reem-kufi text-lg px-8 text-center">
                "Good if your consistent"
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="flex flex-col md:flex-row w-full md:w-3/5 mt-6 mx-auto items-center justify-center">
        <h1 className="text-center text-yellow-400 text-4xl mr-7">★★★★★</h1>
        <p className="font-abo-one text-center text-xl md:text-3xl">
          Over 100+ Reviewers <span className="text-red-500">Love</span> Sale
          Sight
        </p>
      </div>
    </div>
  );
};

export default Reviews;
