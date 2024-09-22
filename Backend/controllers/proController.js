import promodel from "../models/proModel.js";
import fs from 'fs'

//add pro item
const addpro = async(req,res)=>{
let image_filename=`${req.file.filename}`;
const pro = new promodel({
  name:req.body.name,
  description:req.body.description,
  price:req.body.price,
  category:req.body.category,
  image:image_filename

})
try{
   await pro.save();
   res.json({success:true,message:"product added"})
} catch(error){
console.log(error)
res.json({success:false,message:"Error"})
}

}
//all pro list
const listpro = async (req,res)=>{
  try{
    const pro = await promodel.find({});
    res.json({success:true,data:pro})
  } catch (error){
    console.log(error);
    res.json({success:false,message:"Error"})
  }
}
//remove food item
const removepro= async (req,res)=>{
  try{
    const pro= await  promodel.findById(req.body.id);
    fs.unlink(`uploads/${pro.image}`,()=>{})
    await promodel.findByIdAndDelete(req.body.id);
    res.json({success:true,message:"product removed"})
  } catch(error){
    console.log(error);
    res.json({success:false,message:"Error"})


  }

}
export {addpro,listpro,removepro}