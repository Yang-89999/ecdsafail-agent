// A source_edits table (bound to one stream by length + SHA3-256) applied to a stream that later local passes derived
// from that stream. `prepare` checks the binding on the bound stream, as source_edits::run does, and keeps K ops of
// context on each side of every edit; `run` finds each edit's site in the derived stream by its old ops and that
// context (nearest to the expected position), and skips an edit whose site the later passes changed.
use crate::circuit::{Op,OperationType as K};
use sha3::{Digest,Sha3_256};
const CTX:usize=24;
fn rec(o:&Op)->[u64;7]{[o.kind as u64,o.q_control2.0,o.q_control1.0,o.q_target.0,o.c_target.0,o.c_condition.0,o.r_target.0]}
fn mk(v:&[u64])->Op{let kinds=[K::Neg,K::Register,K::AppendToRegister,K::BitInvert,K::BitStore0,K::BitStore1,K::X,K::Z,K::CX,K::CZ,K::Swap,K::R,K::Hmr,K::CCX,K::CCZ,K::PushCondition,K::PopCondition,K::DebugPrint];let mut o=Op::empty();o.kind=kinds[v[0]as usize];o.q_control2.0=v[1];o.q_control1.0=v[2];o.q_target.0=v[3];o.c_target.0=v[4];o.c_condition.0=v[5];o.r_target.0=v[6];o.validate();o}
pub(super) struct Edit{i:usize,old:Vec<[u64;7]>,new:Vec<Op>,pre:Vec<[u64;7]>,post:Vec<[u64;7]>}
pub(super) fn prepare(v:&[Op],table:&str)->Vec<Edit>{
    let mut lines=table.lines();let h:Vec<&str>=lines.next().unwrap().split_whitespace().collect();
    assert_eq!(v.len(),h[0].parse::<usize>().unwrap(),"rebind: bound stream length");
    let mut hash=Sha3_256::new();for o in v{for x in rec(o){hash.update(x.to_le_bytes());}}
    assert_eq!(format!("{:x}",hash.finalize()),h[1],"rebind: bound stream hash");
    let mut out=Vec::new();let mut at=0;
    for l in lines{let a:Vec<u64>=l.split_whitespace().map(|s|s.parse().unwrap()).collect();let(i,n,z)=(a[0]as usize,a[1]as usize,a[2]as usize);
        assert!(i>=at);assert_eq!(a.len(),3+7*(n+z));
        let old:Vec<[u64;7]>=(0..n).map(|j|{let r=rec(&v[i+j]);assert_eq!(r[..],a[3+7*j..3+7*j+7]);r}).collect();
        let pre=v[i.saturating_sub(CTX)..i].iter().map(rec).collect();let post=v[i+n..(i+n+CTX).min(v.len())].iter().map(rec).collect();
        out.push(Edit{i,old,new:a[3+7*n..].chunks(7).map(mk).collect(),pre,post});at=i+n;}
    out}
fn fits(c:&[Op],j:usize,e:&Edit)->bool{
    let n=e.old.len();if j<e.pre.len()||j+n+e.post.len()>c.len(){return false;}
    (0..n).all(|k|rec(&c[j+k])==e.old[k])&&(0..e.pre.len()).all(|k|rec(&c[j-e.pre.len()+k])==e.pre[k])&&(0..e.post.len()).all(|k|rec(&c[j+n+k])==e.post[k])}
pub(super) fn run(c:Vec<Op>,edits:&[Edit])->Vec<Op>{
    let mut out=Vec::with_capacity(c.len()+edits.iter().map(|e|e.new.len()).sum::<usize>());let(mut at,mut drift,mut done)=(0usize,0isize,0usize);
    for e in edits{
        let want=((e.i as isize+drift).max(at as isize)as usize).min(c.len());
        let mut hit=None;
        if e.i==0&&e.old.is_empty(){hit=Some(0);}else{
            let mut r=0usize;loop{let up=want+r;let dn=want.checked_sub(r).filter(|&d|d>=at);
                if up<=c.len()&&fits(&c,up,e){hit=Some(up);break;}
                if let Some(d)=dn{if r>0&&fits(&c,d,e){hit=Some(d);break;}}
                if up>c.len()&&dn.is_none(){break;}r+=1;}}
        match hit{Some(j)=>{out.extend_from_slice(&c[at..j]);out.extend_from_slice(&e.new);at=j+e.old.len();drift=j as isize-e.i as isize;done+=1;}
            None=>eprintln!("rebind: edit at {} (replaces {} op(s)) skipped: its site changed in the later passes",e.i,e.old.len())}}
    out.extend_from_slice(&c[at..]);
    eprintln!("rebind: {}/{} edits applied",done,edits.len());
    out}
