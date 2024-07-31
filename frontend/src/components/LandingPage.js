import React, { useEffect, useRef } from "react";
import Hero from "./Hero";
import BasicDataTable from "./BasicDataTable";
import Reviews from "./Reviews";
import SimpleCTA from "./SimpleCTA";
import SaleSightFlow from "./SaleSightFlow";
import CaseStudy from "./CaseStudy";
import AiExample from "./AiExample";
import LiveDemo from "./LiveDemo";
import SandwichComparsion from "./SandwichComparsion";
import LocomotiveScroll from "locomotive-scroll";
import "locomotive-scroll/dist/locomotive-scroll.css";

const LandingPage = () => {
  const scrollRef = useRef(null);
  const scrollInstanceRef = useRef(null);

  useEffect(() => {
    // Initialize Locomotive Scroll
    scrollInstanceRef.current = new LocomotiveScroll({
      el: scrollRef.current,
      smooth: true,
    });

    // Clean up on unmount
    return () => {
      if (scrollInstanceRef.current) scrollInstanceRef.current.destroy();
    };
  }, []);

  const scrollToSection = (id) => {
    const section = document.getElementById(id);
    if (section && scrollInstanceRef.current) {
      scrollInstanceRef.current.scrollTo(section, {
        offset: 100, // Adjust as needed
        duration: 3, // Adjust duration as needed
      });
    }
  };

  return (
    <>
      <div ref={scrollRef} id="scroll-container" data-scroll-container>
        <div data-scroll-section>
          <Hero scrollToSection={scrollToSection} />
          <BasicDataTable />
          <Reviews />
          <SimpleCTA scrollToSection={scrollToSection} />
          <SaleSightFlow />
          <CaseStudy />
          <AiExample />
          <LiveDemo />
          <SandwichComparsion />
        </div>
      </div>
    </>
  );
};

export default LandingPage;
