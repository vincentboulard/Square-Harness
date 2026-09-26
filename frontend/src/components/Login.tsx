import { useState, type FormEvent } from 'react'
import { login } from '../store'
import { ErrorNote } from './common'
import { BrandMark } from './Shell'

export function Login() {
  const [token, setToken] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      await login(token.trim())
    } catch (reason) {
      setError((reason as Error).message)
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="login paper">
      <form className="login-card" onSubmit={submit}>
        <BrandMark size={56} />
        <h1>Connect this device</h1>
        <p>Open the address that <code>square-harness --gui</code> printed in your terminal, or paste the access token saved in <code>.mathagent/gui-token</code> inside your workspace.</p>
        <label className="field">
          <span className="field-label">Access token</span>
          <input type="password" autoComplete="off" spellCheck={false} value={token}
            onChange={(event) => setToken(event.target.value)} autoFocus />
        </label>
        <ErrorNote>{error}</ErrorNote>
        <button type="submit" className="btn btn-primary btn-block" disabled={busy || !token.trim()}>Connect</button>
      </form>
    </div>
  )
}
