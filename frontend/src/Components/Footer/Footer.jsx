import React from 'react'
import './Footer.css'
import { assets } from '../../assets/assets'
const Footer = () => {
  return (
    <div className='footer' id='footer'>
      <div className='footer-content'>
        <div className="footer-content-left">
          <img src={assets.logo} alt="" />
          <p>E-Store websites offer convenient shopping, a vast product selection, competitive prices, easy price comparison, and 24/7 accessibility from anywhere.</p>
          <div className="footer-social_icon">
            <img src={assets.facebook_icon} alt="" />
            <img src={assets.twitter_icon} alt="" />
            <img src={assets.linkedin_icon} alt="" />
          </div>

        </div>
        <div className="footer-content-center">
          <h2>COMPANY</h2>
          <ul>
            <li>Home</li>
            <li>About us</li>
            <li> Delivery</li>
            <li> Privacy Policy</li>
          </ul>

        </div>
        <div className="footer-content-right">
        <h2>GET IN TOUCH</h2>
        <ul>
          <li>+977-9811765436</li>
          <li> contact @E-STORE.COM</li>
        </ul>
        </div>


      </div>
      <hr />
      <p className='footer-copyright'> Copyright 2024 @ E-STORE - ALL RIGHT RESERVED</p>




    </div>
  )
}

export default Footer