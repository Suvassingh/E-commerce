import mongoose from "mongoose";
 export const connectDB = async()=>{
  await mongoose.connect('mongodb+srv://ss5138275:Renudinesh@cluster0.gdppkcv.mongodb.net/pro-del').then(()=>console.log("DB connected"));
}