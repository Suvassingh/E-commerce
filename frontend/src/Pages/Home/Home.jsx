// eslint-disable-next-line no-unused-vars
import React, { useState } from 'react'
import './Home.css'
import Header from '../../Header/Header'
import ExploreMenu from '../../Components/Navbar/ExploreMenu/ExploreMenu'
import ItemDisplay from '../../Components/ItemDisplay/ItemDisplay'
import Appdownload from '../../Components/AppDownload/Appdownload'
const Home = () => {
  const [ category,setCategory]=useState("All")
  return (
    <div>
      <Header/>
      <ExploreMenu category={category} setCategory={setCategory}/>
      <ItemDisplay category={category} />
      <Appdownload/>
      




    </div>
  )
}

export default Home