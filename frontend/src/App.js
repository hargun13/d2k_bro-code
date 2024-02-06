import React, {useState, useEffect} from 'react';
import {Routes, Route} from 'react-router-dom'
import { AuthContextProvider } from './FirebaseAuthContext/AuthContext';

import Landing from './Landing/Landing/Landing';
import Home from './Blog/Home';
import Community from './Community/Community'
import Login from './Login_Signup/Login';
import Signup from './Login_Signup/Signup';

import Sidebar from './components/Sidebar'
import MainDash from './Main/MainDash';
import PlaceExtracted from './Main/Maps/PlaceExtracted'
import UserProfile from './Main/Profile/UserProfile'
import Exercises from './Main/Exercise/Exercises';
import ExerciseMain from './Main/Exercise/ExerciseMain'; //exercises

function App() {

  const [login, setLogin] = useState(false);
  useEffect(()=>{}, [login]) 

  return (
    <div className="h-full w-full">
      <AuthContextProvider>
        {login && <Sidebar setLogin={setLogin} />}
        <Routes>
          <Route path='/' element={<Landing/>} />
          <Route path='/Blog' element={<Home/>} />
          <Route path='/Community' element={<Community/>} />
          <Route path='/Login' element={<Login setLogin={setLogin} />} />
          <Route path='/SignUp' element={<Signup setLogin={setLogin} />} />
          <Route path='/Dashboard' element={<MainDash/>} />
          <Route path='/final' element={<PlaceExtracted/>} />
          <Route path='/Profile' element={<UserProfile/>} />
          <Route path='/exercise' element={<Exercises />} />
          <Route path='/exercise/:exerciseId' element={<ExerciseMain />} />
        </Routes>
      </AuthContextProvider>
    </div>
  );
}

export default App;
