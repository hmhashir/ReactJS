import React from 'react'
import card from './components/card'
import navbar from './components/navbar'

const App = () => {
  return (
    <>
    <div>App</div>
    {card()}
    {navbar()}
    </>
  )
}

export default App