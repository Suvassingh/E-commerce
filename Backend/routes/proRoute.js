import express from "express"
import { addpro ,listpro,removepro} from "../controllers/proController.js"
import multer from "multer"
const proRouter = express.Router();
//image storage engine
const storage =multer.diskStorage({
  destination:"uploads",
  filename:(req,file,cb)=>{
    return cb(null,`${Date.now()}${file.originalname}`)
  }
})
const upload =multer({storage:storage})
proRouter.post("/add",upload.single("image"),addpro)

proRouter.get("/list",listpro)

proRouter.post("/remove",removepro)

export default proRouter;
