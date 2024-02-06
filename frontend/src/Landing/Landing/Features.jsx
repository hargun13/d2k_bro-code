import React from 'react'
import { BsFillGeoAltFill } from "react-icons/bs";
import { FaMapLocation } from "react-icons/fa6";
import { AiOutlineSafetyCertificate } from "react-icons/ai";
import { TbMoneybag } from "react-icons/tb";
// import nhf from './no_hidden_fees.png'

const Features = () => {


    const bag2 = "https://assets.website-files.com/62cc07ca0720bd63152e1799/62cd16b4a5613c06cf9a0ff4_line-bg.svg";

    const data = [
        {
            logo: <BsFillGeoAltFill size={40}/>,
            heading: "Dynamic Geolocation Insights",
            content: "Engage users with dynamic and interactive geolocation insights, transforming location data into an engaging and enjoyable exploration experience."
        },
        {
            logo: <FaMapLocation size={40}/>,
            heading: "Real-time Location Tracking",
            content: "Provide users with real-time geospatial updates and tracking tools, empowering them to explore and analyze textual data based on specific geographic locations."
        },
        {
            logo: <TbMoneybag size={40}/>,
            heading: "Rewarding Geolocation Achievements",
            content: "Create a sense of accomplishment and motivation through challenges, rewarding users for extracting valuable geolocation insights and reaching data milestones."
        },
        {
            logo: <AiOutlineSafetyCertificate size={40}/>,
            heading: "Geospatial Certificates and Badges",
            content: "Offer tangible recognition of users' geospatial achievements with certificates and badges upon successfully extracting and analyzing geolocation information."
        },
    ];
    
  return (
    <div style={{backgroundImage: `url(${bag2})`, backgroundSize:'cover'}} className='h-auto flex flex-col items-center justify-center py-5'>
        
        <h1 className='text-5xl  font-bold text-center py-5'>Fun and Exciting Features</h1>
        <p className='md:text-xl text-lg italic font-semibold text-gray-500 text-center my-5 px-10'>Unlock valuable geolocation insights effortlessly. With our app, <br/> you can extract and analyze location data seamlessly.</p>

        <div className='grid md:grid-cols-4 grid-cols-1 place-items-center items-center justify-center m-10 '>
            
            {data.map((data) =>(

            <div className='mx-5 p-6 bg-white rounded-2xl transition duration-700 hover:-translate-y-3 hover:bg-[#1976D2] hover:text-white group shadow-2xl'>

                <span>{data.logo}</span>
                <h1 className='text-2xl font-bold my-5'>{data.heading}</h1>
                <p className=' text-lg font-bold italic text-gray-300 transition duration-700 group-hover:text-white'>{data.content}</p>

            </div>

            ))}

        </div>


    </div>
  )
}

export default Features