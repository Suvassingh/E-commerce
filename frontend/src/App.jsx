import React, { useState } from 'react'
import Navbar from './Components/Navbar/Navbar'
import { Route, Routes } from 'react-router-dom'
import Home from './Pages/Home/Home'
import Cart from './Pages/Cart/Cart'
import PlaceOrder from './Pages/PlaceOrder/PlaceOrder'
import Footer from './Components/Footer/Footer'
import LoginPopup from './Components/LoginPopup.jsx/LoginPopup'
import Verify from './Pages/verify/Verify'
import Myorder from './Pages/Myorder/Myorder'
//import Verify from './Pages/verify/Verify'



const App = () => {
  const[showLogin,setShowLogin]=useState(false)
  return (
    <>
    {showLogin?< LoginPopup setShowLogin={setShowLogin}/>:<></>}
    <div className='app'>
    <Navbar setShowLogin={setShowLogin}/>
    <Routes>
      <Route path='/' element={<Home/>}/>
      <Route path='/cart' element={<Cart/>}/>
      <Route path='/order' element={<PlaceOrder/>}/>
      <Route path='/Verify' element={<Verify/>}/>
      <Route path='/myorder' element={<Myorder/>}/>
      


    </Routes>


  </div>
  <Footer/>
  </>
  )
}

export default App