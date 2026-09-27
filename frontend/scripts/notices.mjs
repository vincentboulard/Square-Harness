// Write the licences of every production dependency next to the built bundle.
// The minified bundle keeps no licence comments, and MIT/ISC/OFL require the notices.
import { execFileSync } from 'node:child_process'
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

const tree = JSON.parse(execFileSync('npm', ['ls', '--omit=dev', '--all', '--json', '--long'], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 }))
const found = new Map()
;(function walk(node) {
  for (const dependency of Object.values(node.dependencies || {})) {
    if (dependency.path && !found.has(dependency.path)) {
      found.set(dependency.path, dependency)
      walk(dependency)
    }
  }
})(tree)

const sections = []
for (const [path, dependency] of [...found.entries()].sort((a, b) => a[1].name.localeCompare(b[1].name))) {
  const manifest = JSON.parse(readFileSync(join(path, 'package.json'), 'utf8'))
  const file = readdirSync(path).find((name) => /^(licen[cs]e|copying)(\.|$)/i.test(name))
  const text = file ? readFileSync(join(path, file), 'utf8').trim() : `License: ${manifest.license || 'see package'}`
  sections.push(`${manifest.name} ${manifest.version} (${manifest.license || 'unknown'})\n${'-'.repeat(60)}\n${text}`)
}
const out = '../mathagent/gui/static/THIRD-PARTY-NOTICES.txt'
if (!existsSync('../mathagent/gui/static')) throw new Error('Build the interface first')
writeFileSync(out, [
  'Third-party software bundled in the Square Harness visual interface.',
  'The interface also bundles Square Concrete and Square Typewriter, subset and renamed from',
  'Computer Modern Unicode (SIL Open Font License 1.1): see fonts/OFL-square-concrete.txt.',
  '', ...sections, '',
].join('\n\n'))
console.log(`notices: ${sections.length} packages → ${out}`)
