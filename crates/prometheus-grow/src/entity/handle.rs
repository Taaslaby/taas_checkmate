//! BioDynaMo `core/agent/agent_handle.h` + `new_agent_event.h` port

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityHandle(pub usize);

#[derive(Clone, Debug)]
pub struct NewEntityEvent { pub parent: EntityHandle, pub pos: crate::Vec3 }

impl NewEntityEvent {
    pub fn new(parent: usize, pos: crate::Vec3) -> Self { NewEntityEvent { parent: EntityHandle(parent), pos } }
}
