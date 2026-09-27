import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/atkinson-hyperlegible-next'
import './theme.css'
import './shell.css'
import './content.css'
import { App } from './App'
import { applyTheme, readTheme, start } from './store'

applyTheme(readTheme())
start()
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
