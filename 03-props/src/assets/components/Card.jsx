import React from 'react'

const ard = (props) => {

    return (
        <div className="card">
            <img src={props.img} alt="" />
            <h1>{props.user},{props.age}</h1>
            <p>Lorem ipsum dolor sit amet consectetur adipisicing elit.</p>
            <button>View Profile</button>
        </div>
    )
}

export default ard