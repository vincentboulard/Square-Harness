import { useEffect, useState } from 'react'
import { ApprovalDialog } from './components/Approval'
import { DropZone } from './components/DropZone'
import { Login } from './components/Login'
import { BrandMark, Rail, Sidebar, TopBar } from './components/Shell'
import { MODES } from './modes'
import { useRoute, type Route } from './router'
import { useApp } from './store'
import { ChatView } from './views/ChatView'
import { ProofNew } from './views/ProofNew'
import { ProofView } from './views/ProofView'
import { ResearchNew } from './views/ResearchNew'
import { ResearchView } from './views/ResearchView'
import { WriteupNew } from './views/WriteupNew'
import { WriteupView } from './views/WriteupView'

function View({ route }: { route: Route }) {
  const info = MODES[route.mode]
  if (info.kind === 'proof') return route.id ? <ProofView key={route.id} id={route.id} tab={route.tab} /> : <ProofNew />
  if (info.kind === 'chat' || info.kind === 'free') {
    return <ChatView key={route.mode + (route.id || '')} mode={route.mode as 'critic' | 'explore' | 'free'} id={route.id} />
  }
  if (info.kind === 'writeup') return route.id ? <WriteupView key={route.id} id={route.id} tab={route.tab} /> : <WriteupNew />
  return route.id
    ? <ResearchView key={route.id} id={route.id} tab={route.tab} />
    : <ResearchNew key={route.mode} kind={route.mode as 'literature' | 'referee'} />
}

export function App() {
  const app = useApp()
  const route = useRoute()
  const [drawer, setDrawer] = useState(false)
  useEffect(() => setDrawer(false), [route.mode, route.id])
  if (app.auth === 'checking') {
    return <div className="splash paper"><BrandMark size={56} /></div>
  }
  if (app.auth === 'needed') return <Login />
  return (
    <div className="app">
      <div className={'nav' + (drawer ? ' nav-open' : '')}>
        <Rail route={route} />
        <Sidebar route={route} />
      </div>
      {drawer && <div className="scrim" onClick={() => setDrawer(false)} />}
      <main className="main">
        <TopBar title={MODES[route.mode].label} onMenu={() => setDrawer(true)} />
        {app.connection === 'reconnecting' && (
          <div className="connection" role="status">Reconnecting to the harness. Saved work is safe; live updates resume when it answers.</div>
        )}
        <View route={route} />
      </main>
      <ApprovalDialog />
      <DropZone />
    </div>
  )
}
