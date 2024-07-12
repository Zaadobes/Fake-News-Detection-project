import "./App.css"
import fns_extract from './images/fns_extract.png'
import { RiChatNewLine,RiHistoryLine, RiSettings2Line, RiChatUploadLine} from "react-icons/ri";
import { useNavigate } from "react-router-dom";
import {auth} from './firebase.jsx'
import { signOut } from 'firebase/auth';

const Home = () => {
    const move = useNavigate()

    const LogOut = async(e)=> { 
        e.preventDefault()
        try{
             await signOut(auth)
             console.log("out")
             move("/")
            } catch(err){
                console.log(err.message)
                console.log(err.code)
            }
          } 
      
    return ( 
        <div className="home">
            <div className="Menu" >
            <img id="logo1" src={fns_extract} alt="logo"/> 

            {/* <p>{username}</p> */}
            <ul>
                <li><RiChatNewLine/>New Chat</li>
                <li><RiHistoryLine/>History</li>
                <li><RiSettings2Line/>Settings</li>
            </ul>
            <form>
            <button className="signOut" type="button" onClick={LogOut} >Sign Out</button>
            </form>
            </div>


            <div className="MainSearch">
                <input className="search" type="text" placeholder="Enter article here.." />
                <button type="submit" className="upload" ><RiChatUploadLine /></button>
            </div>
            
        </div>
        
     );
}
 
export default Home;