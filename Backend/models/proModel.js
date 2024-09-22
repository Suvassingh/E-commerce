import mongoose from "mongoose";
const proSchema=new mongoose.Schema({
  name:{type: String,required:true},
  description:{type:String,required:true},
  price:{type:Number,required:true},
  image:{type:String,required:true},
  category:{type:String,required:true}
})
const promodel= mongoose.models.pro || mongoose.model("pro",proSchema);
export default promodel;
