import axios from "axios";
import { createContext, useEffect, useState } from "react";

export const StoreContext= createContext(null)
const StoreContextProvider=(props)=>{
 
  const [cartItem,setCartItem]=useState({});
  const url = "http://localhost:5000";
  const [token,setToken] = useState("");
  const [item_list,setItemList] = useState([])
    const addToCart= async (itemId)=>{
      if (!cartItem[itemId]){
        setCartItem((prev)=>({...prev,[itemId]:1}))

      }
      else{
        setCartItem((prev)=>({...prev,[itemId]:prev[itemId]+1}))
      }
      if (token){
        await axios.post(url+"/api/cart/add",{itemId},{headers:{token}})


      }
    }
    const removeFromCart= async(itemId)=>{
      setCartItem((prev)=>({...prev,[itemId]:prev[itemId]-1}));
      if (token){
        await axios.post(url+"/api/cart/remove",{itemId},{headers:{token}})

      }
    }
    const getTotalCartAmount=()=>{
      let totalAmount=0;
      for (const item in cartItem)
        {
          if (cartItem[item]>0){
            let itemInfo=item_list.find((product)=>product._id===item);
          totalAmount+= itemInfo.price*cartItem[item];

          }
        }
        return totalAmount;
    }
    const fetchItemList= async ()=>{
      const response = await axios.get(url+"/api/pro/list");
      setItemList(response.data.data)
    }
    const loadCartData = async ( token)=>{
      const response = await axios.post ( url+"/api/cart/get",{},{headers:{token}});
      setCartItem ( response.data.cartData);

    }
    useEffect (()=>{
     
      async function loadData(){
        await fetchItemList();
        if (localStorage.getItem("token")){
          setToken(localStorage.getItem("token"));
           await loadCartData(localStorage.getItem("token"));
        }
      }
      loadData();
    },[])
  const contextValue={
    item_list,
    cartItem,
    setCartItem,
    addToCart,
    removeFromCart,
    getTotalCartAmount,
    url,
    token,
    setToken


  }
  return (
    <StoreContext.Provider value={contextValue}>
      {props.children}
    </StoreContext.Provider>
  )
}
export default StoreContextProvider