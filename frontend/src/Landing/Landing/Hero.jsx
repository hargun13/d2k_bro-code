import React, {useState} from 'react'
import Navbar from './Navbar'
import img1 from './hero/1.png'

import { FaCheckCircle } from "react-icons/fa";
import { FaArrowCircleRight } from "react-icons/fa";
import { FaArrowCircleLeft } from "react-icons/fa";


const Carousel = ({ textData }) => {
    const [currentIndex, setCurrentIndex] = useState(0);
  
    const handleNext = () => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % textData.length);
    };
  
    const handlePrev = () => {
      setCurrentIndex((prevIndex) => (prevIndex - 1 + textData.length) % textData.length);
    };
  
    return (
        <div className="text-xs w-[80%] flex items-center justify-between ">
            <div className="w-[75%]">
                <p >{textData[currentIndex]}</p>
            </div>
            <div className="flex w-[20%] items-center justify-between">
                <FaArrowCircleLeft onClick={handlePrev} size={30} className='text-[#A0BDFF] cursor-pointer'/>
                <FaArrowCircleRight onClick={handleNext} size={30} className='text-[#A0BDFF] cursor-pointer'/>
            </div>
        </div>
    );
  };


const Hero = () => {

    const text = [
        "Unlock personalized geolocation insights designed to evolve alongside your journey, ensuring a unique and impactful exploration of diverse locations.",
        "Maintain command with real-time geospatial analytics, empowering you to make informed decisions and seize opportunities tied to geographical well-being.",
        "Engage with a dynamic community of location enthusiasts and geospatial experts, encouraging collaboration, discussions, and shared moments for an enriched understanding of global positions"
    ];
    
  return (
    <div className='h-auto bg-white pb-20 ' >
        <Navbar />

        <div className='md:flex w-full items-center justify-center mt-5 text-black'>

            <div className='md:w-[60%] px-16'>
                
                <div className='flex items-center justify-start text-[10px] gap-3'>
                    <p className='bg-[#F0F1EB] py-1 px-3 rounded-full font-bold'>GEOGRAPHY</p>
                    <p className='bg-[#F0F1EB] py-1 px-3 rounded-full font-bold'>GEOLOCATION</p>
                </div>

                <h1 className='md:text-[45px] text-5xl font-semibold py-3 '>Discover, Map, Analyze!</h1> 
                <h1 className='md:text-[45px] text-5xl font-semibold pt-3 -ml-4'><span className='bg-[#A0BDFF] px-4 rounded-full'>GeoLoc Insights</span> Unleashed!</h1>

                <p className="text-sm py-10 pr-40 font-bold my-5">
                    Embark on a journey where text transforms into an exciting exploration of locations. Explore, learn, and thrive with our engaging platform designed to make geolocation extraction enjoyable and insightful.
                </p>

                <div className='flex items-center gap-7 uppercase text-xs font-extrabold'>
                    <p className='flex items-center justify-start gap-1'><FaCheckCircle size={20} className='text-[#A0BDFF]'/>Location-Based Data</p>
                    <p className='flex items-center justify-start gap-1'><FaCheckCircle size={20} className='text-[#A0BDFF]'/>Geospatial Analysis</p>
                    <p className='flex items-center justify-start gap-1'><FaCheckCircle size={20} className='text-[#A0BDFF]'/>Visualize Geographic Insights</p>
                </div>

                
                <hr className=' border border-[#F0F1EB] my-3 w-[80%]'/>

                <Carousel textData={text}/>
                
            </div>

            <div className='md:w-[40%] flex justify-center'>
                    
                <img src={img1} alt="hero-1" />

            </div>


        </div>


    </div>
  )
}

export default Hero