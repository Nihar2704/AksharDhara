import { useState } from 'react'
import SpeechTranslator from './SpeechTranslator'


function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <SpeechTranslator />
    </>
  )
}

export default App
