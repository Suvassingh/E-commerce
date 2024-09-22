import React from 'react'
import './AppDownload.css'
import { assets } from '../../assets/assets'
const Appdownload = () => {
  return (
    <div className='app-download' id ='app-download'>
      <p> For Better Experience Download <br/>E-STORE App</p>
      <div className="app-download-platform">
        <img src={assets.play_store} alt="" />
        <img src={assets.app_store} alt="" />
      </div>


    </div>
  )
}

export default Appdownload