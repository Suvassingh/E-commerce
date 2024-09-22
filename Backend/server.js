import express from "express"
import cors from "cors"

import proRouter from "./routes/proRoute.js"
import { connectDB } from "./config/db.js"
import userRouter from "./routes/userRoute.js"
//import userRouter from "./routes/userRoute.js"
import 'dotenv/config'
import cartRouter from "./routes/cartRoute.js"
import orderRouter from "./routes/orderRoute.js"
const app=express()
const port = 5000
// middleware
app.use(express.json())
app.use(cors())
//db connection
connectDB();
//api endpoints
app.use("/api/pro",proRouter)
app.use("/images",express.static('uploads'))
app.use("/api/user",userRouter)
app.use("/api/cart",cartRouter)
app.use("/api/order",orderRouter)


app.get("/",(req,res)=>{
  res.send("API Working")
})
app.listen(port,()=>{
  console.log(`server started on http://localhost:${port}`)
})
//mongodb+srv://ss5138275:Renudinesh@cluster0.gdppkcv.mongodb.net/?