import React from 'react'
import {Bookmark} from 'lucide-react'

const Card = () => {
  return (
    <div className="card">
        <div>
          <div className="top">
          <img src="https://tse4.mm.bing.net/th/id/OIP.xUrX3S3mxDhB4X4L5ZlU5QHaHa?rs=1&pid=ImgDetMain&o=7&rm=3" alt="" />
          <button>Save <Bookmark size={12} /> </button>
        </div>

        <div className="center">
          <h3>Amazon <span>5 days ago</span> </h3>
          <h2>Senior UI/UX Designer</h2>
          <div className='tag'>
            <h4>Part Time</h4>
            <h4>Senior Level</h4>
          </div>
        </div>
        </div>
        
        <div className="bottom">
          <div>
            <h3>$180/hr</h3>
            <p>Lahore, Pakistan</p>
          </div>
          <button>Apply Now</button>
        </div>

    </div>
  )
}

export default Card